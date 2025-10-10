import os
import logging
import argparse
import torch
import numpy as np   
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------
# Configurações
# -------------------
CLASSES = ["Nenhuma", "Leve", "Severa"]

# Thresholds ideais por classe (baseado nos seus testes)
CLASS_THRESHOLDS = {
    "Nenhuma": 0.7,
    "Leve": 0.3,
    "Severa": 0.6
}

# -------------------
# Funções
# -------------------

def load_dataset(file_path: str, text_column: str, label_column: str):
    """Carrega o dataset e divide em treino, validação e teste (70/15/15)"""
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
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df), Dataset.from_pandas(test_df)


def preprocess_data(train_ds, val_ds, test_ds, tokenizer, text_column, label_column, max_length=128):
    """Tokeniza textos e codifica labels para PyTorch"""
    
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
    """Calcula métricas de avaliação usando argmax puro (sem thresholds)"""
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
    """Treina o modelo BERT"""
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


def evaluate_model(model, test_ds):
    """Avalia o modelo no conjunto de teste"""
    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    results = trainer.evaluate(test_ds)
    logger.info(f"Resultados da avaliação: {results}")
    return results


def predict_with_class_thresholds(model, tokenizer, texts):
    """Predição aplicando threshold específico para cada classe"""
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        # Verifica se alguma probabilidade passa no threshold da própria classe
        passed_classes = [cls for i, cls in enumerate(CLASSES) if probs[i] >= CLASS_THRESHOLDS[cls]]

        if passed_classes:
            # Seleciona a classe com maior probabilidade dentre as que passaram no threshold
            idx = np.argmax([probs[CLASSES.index(cls)] for cls in passed_classes])
            pred_class = passed_classes[idx]
        else:
            pred_class = "Nenhuma"  # default se nenhuma classe passar no threshold

        predictions.append(pred_class)
    return predictions


# -------------------
# Main
# -------------------

def main():
    parser = argparse.ArgumentParser(description="Treinamento BERTimbau multiclasse")
    parser.add_argument('--data_path', type=str, default='Treinamento/LI/LI_processado.csv')
    parser.add_argument('--text_column', type=str, default='Frase')
    parser.add_argument('--label_column', type=str, default='Intensidade')
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='neuralmind/bert-base-portuguese-cased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_ds, val_ds, test_ds = load_dataset(args.data_path, args.text_column, args.label_column)
    train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds, tokenizer, args.text_column, args.label_column, args.max_length)

    output_dir_task = os.path.join(args.output_dir, "bertimbau_multiclass")
    os.makedirs(output_dir_task, exist_ok=True)

    model = train_model(args.model_name, train_ds, val_ds, output_dir_task, args.epochs, args.batch_size, args.learning_rate)
    results = evaluate_model(model, test_ds)

    # Salva resultados
    results_file = os.path.join(output_dir_task, "evaluation_results.txt")
    with open(results_file, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    logger.info(f"Treinamento concluído. Modelo salvo em {output_dir_task}")

if __name__ == "__main__":
    main()
