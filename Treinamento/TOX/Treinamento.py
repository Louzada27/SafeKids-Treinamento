#!/usr/bin/env python3
"""
Treinamento BERTimbau TOX puro:
- Sem class weight
- Sem threshold
- Sem early stopping
- Labels como inteiros (0,1,2,3)
"""

import os
import logging
import torch
import argparse
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import matplotlib.pyplot as plt

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------- Configurações -------------------
CLASSES = [0, 1, 2, 3]

# ------------------- Funções -------------------
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
    df[label_column] = df[label_column].astype(int)  # labels como int

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

    def encode_labels(batch):
        batch["labels"] = [int(label) for label in batch[label_column]]
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
    report = classification_report(labels, preds, labels=CLASSES, output_dict=True, zero_division=0)
    metrics = {}
    for cls in CLASSES:
        metrics[f"eval_{cls}_precision"] = report[str(cls)]["precision"]
        metrics[f"eval_{cls}_recall"] = report[str(cls)]["recall"]
        metrics[f"eval_{cls}_f1"] = report[str(cls)]["f1-score"]
    return metrics

def train_model(model_name, train_ds, val_ds, output_dir, epochs=5, batch_size=16, lr=5e-5):
    logger.info(f"Treinando modelo {model_name}")
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
        save_total_limit=2,
        load_best_model_at_end=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    return model

# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser(description="Treinamento BERTimbau TOX puro")
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

    output_dir_task = os.path.join(args.output_dir, "Tox")
    os.makedirs(output_dir_task, exist_ok=True)

    model = train_model(args.model_name, train_ds, val_ds, output_dir_task, args.epochs, args.batch_size, args.learning_rate)

    # Predição final puro argmax
    texts = df[args.text_column].tolist()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=args.max_length)
    with torch.no_grad():
        logits = model(**inputs).logits
        preds = logits.argmax(-1).tolist()
    true_labels = df[args.label_column].tolist()

    print(classification_report(true_labels, preds, target_names=[str(c) for c in CLASSES]))
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(c) for c in CLASSES])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Matriz de Confusão - Modelo BERTimbau TOX")
    plt.show()

    logger.info(f"Treinamento concluído. Modelo salvo em {output_dir_task}")

if __name__ == "__main__":
    main()
