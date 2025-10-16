import os
import logging
import torch
import argparse
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------
# Configurações
# -------------------
CLASSES = ["Nenhuma", "Leve", "Moderado", "Severo"]
THRESHOLD_LEVE = 0.45  # Threshold aplicado somente à classe Leve

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

    metrics = {"eval_accuracy": acc}
    f1_weighted = np.average(f1, weights=support)
    metrics["eval_f1_weighted"] = f1_weighted

    for i, cls in enumerate(CLASSES):
        metrics[f"eval_{cls}_precision"] = precision[i]
        metrics[f"eval_{cls}_recall"] = recall[i]
        metrics[f"eval_{cls}_f1"] = f1[i]
    return metrics

def train_model(model_name, train_ds, val_ds, output_dir, df, label_column, epochs=8, batch_size=16, lr=5e-5):
    logger.info(f"Treinando modelo {model_name} com peso fixo apenas para classe 'Leve'")

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(CLASSES))

    # Peso fixo apenas para a classe "Leve"
    weights = torch.ones(len(CLASSES), dtype=torch.float)
    idx_leve = CLASSES.index("Leve")
    weights[idx_leve] = 3.0  # peso fixo

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(model.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

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
        metric_for_best_model="eval_f1_weighted",  # corrigido
        load_best_model_at_end=True,
        save_total_limit=2
    )

    trainer = WeightedTrainer(
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

def predict_with_leve_threshold(model, tokenizer, texts, threshold=THRESHOLD_LEVE):
    model.eval()
    predictions = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        idx_leve = CLASSES.index("Leve")
        if probs[idx_leve] >= threshold and probs[idx_leve] > probs.argmax() - 0.05:
            pred_class = "Leve"
        else:
            pred_class = CLASSES[probs.argmax()]

        predictions.append(pred_class)

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Treinamento BERTimbau com class weight e threshold Leve")
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

    output_dir_task = os.path.join(args.output_dir, "bertimbau_tox_leve_weighted")
    os.makedirs(output_dir_task, exist_ok=True)

    model = train_model(args.model_name, train_ds, val_ds, output_dir_task, df, args.label_column, args.epochs, args.batch_size, args.learning_rate)

    # Predição com threshold só na classe "Leve"
    texts = df[args.text_column].tolist()
    preds = predict_with_leve_threshold(model, tokenizer, texts, threshold=THRESHOLD_LEVE)
    true_labels = df[args.label_column].tolist()

    print(classification_report(true_labels, preds, labels=CLASSES))

    # Matriz de confusão
    cm = confusion_matrix(true_labels, preds, labels=CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Matriz de Confusão - Modelo BERTimbau TOX")
    plt.show()

    logger.info(f"Treinamento concluído. Modelo salvo em {output_dir_task}")

if __name__ == "__main__":
    main()
