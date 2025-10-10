import os
import logging
import argparse
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
import numpy as np
from torch import nn, optim

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------
# Configurações
# -------------------
CLASSES = ["Nenhuma", "Leve", "Moderado", "Severo"]

# Thresholds independentes para cada classe (ajuste conforme experimentos)
CLASS_THRESHOLDS = {
    "Nenhuma": 0.56, #ok
    "Leve": 0.5,
    "Moderado": 0.5, #ok
    "Severo": 0.6 #ok
}

# -------------------
# Funções
# -------------------

def load_dataset(file_path: str, text_column: str, label_column: str):
    logger.info(f"Carregando dataset de {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")

    df = df.dropna(subset=[text_column, label_column])
    df[label_column] = df[label_column].astype(str).str.strip()

    logger.info(f"Dataset carregado com {len(df)} exemplos")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[label_column])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[label_column])

    logger.info(f"Divisão - Treino: {len(train_df)}, Validação: {len(val_df)}, Teste: {len(test_df)}")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df), Dataset.from_pandas(test_df), df


def preprocess_data(train_ds, val_ds, test_ds, tokenizer, text_column, label_column, max_length=128):
    def tokenize(batch):
        return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=max_length)
    
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    label_map = {label: i for i, label in enumerate(CLASSES)}
    def encode_labels(batch):
        batch["labels"] = [label_map[label] for label in batch[label_column]]
        return batch

    train_ds = train_ds.map(encode_labels, batched=True)
    val_ds = val_ds.map(encode_labels, batched=True)
    test_ds = test_ds.map(encode_labels, batched=True)

    columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)

    return train_ds, val_ds, test_ds


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    acc = accuracy_score(labels, preds)

    per_class_metrics = {}
    total_support = sum(support)
    weighted_precision = weighted_recall = weighted_f1 = 0

    for i, cls in enumerate(CLASSES):
        per_class_metrics[cls] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i]
        }
        weighted_precision += precision[i] * support[i]
        weighted_recall += recall[i] * support[i]
        weighted_f1 += f1[i] * support[i]

    metrics = {
        "accuracy": acc,
        "precision_mean": weighted_precision / total_support,
        "recall_mean": weighted_recall / total_support,
        "f1": weighted_f1 / total_support
    }
    metrics.update(per_class_metrics)
    return metrics


def train_model(model_name, train_ds, val_ds, output_dir, epochs=5, batch_size=16, lr=5e-5):
    logger.info(f"Treinando modelo {model_name} para {len(CLASSES)} classes")
    
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(CLASSES))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model(output_dir)
    return model


# -------------------
# Calibração de Temperatura
# -------------------
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Cria o parâmetro de temperatura antes de enviar pro device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  #

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        logits = outputs.logits
        return logits / self.temperature

    def set_temperature(self, val_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        logits_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                logits = self.model(**inputs).logits
                logits_list.append(logits)
                labels_list.append(inputs["labels"])
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        logger.info(f"Temperatura calibrada: {self.temperature.item():.4f}")
        return self


def calibrate_model(model, val_ds):
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=32)
    calibrated_model = ModelWithTemperature(model)
    calibrated_model.set_temperature(val_loader)
    return calibrated_model


def predict_with_thresholds(model, tokenizer, texts, thresholds, temperature=1.0):
    model.eval()
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits / temperature, dim=-1).numpy()[0]

        # Aplica thresholds específicos por classe
        pred_class = "Nenhuma"
        for i, cls in enumerate(CLASSES):
            if probs[i] >= thresholds[cls]:
                pred_class = cls
                break
        predictions.append(pred_class)
    return predictions


def evaluate_model(model, test_ds):
    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    results = trainer.evaluate(test_ds)
    logger.info(f"Resultados da avaliação: {results}")
    return results


# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Treinamento BERTimbau multiclasse TOX com calibração de temperatura e thresholds por classe")
    parser.add_argument('--data_path', type=str, default='Treinamento/TOX/TOX_processado.csv')
    parser.add_argument('--text_column', type=str, default='FRASE')
    parser.add_argument('--label_column', type=str, default='Intensidade')
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='neuralmind/bert-base-portuguese-cased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_ds, val_ds, test_ds, df = load_dataset(args.data_path, args.text_column, args.label_column)
    train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds, tokenizer, args.text_column, args.label_column, args.max_length)

    output_dir_task = os.path.join(args.output_dir, "bertimbau_tox_thresholds")
    os.makedirs(output_dir_task, exist_ok=True)

    model = train_model(args.model_name, train_ds, val_ds, output_dir_task, args.epochs, args.batch_size, args.learning_rate)

    # ---- Calibração de temperatura ----
    logger.info("Iniciando calibração de temperatura...")
    calibrated_model = calibrate_model(model, val_ds)
    T_value = calibrated_model.temperature.item()
    with open(os.path.join(output_dir_task, "temperature.txt"), "w") as f:
        f.write(f"{T_value:.4f}")

    # ---- Avaliação final ----
    texts = df[args.text_column].tolist()
    preds = predict_with_thresholds(model, tokenizer, texts, thresholds=CLASS_THRESHOLDS, temperature=T_value)
    true_labels = df[args.label_column].tolist()

    print(classification_report(true_labels, preds, labels=CLASSES))
    results = evaluate_model(model, test_ds)
    results_file = os.path.join(output_dir_task, "evaluation_results.txt")
    with open(results_file, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    logger.info(f"Treinamento e calibração concluídos. Modelo salvo em {output_dir_task}")


if __name__ == "__main__":
    main()
