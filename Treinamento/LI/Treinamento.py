#!/usr/bin/env python3
"""
Treinamento BERTimbau multiclass (classes 0, 1, 2) com:
- Avaliação completa: F1, Recall e Precision
- Ajuste por temperatura (para cálculo das métricas)
- Threshold aplicado apenas na classe 1
- Salva resultados do modelo final
"""

import os
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, BertConfig

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASSES = [0, 1, 2]
TEMPERATURE = 1.0  # para calibrar as probabilidades
CLASS_THRESHOLD = {0: 0.0, 1: 0.50, 2: 0.0}  # Threshold aplicado apenas na classe 1

# -------------------
# Dataset
# -------------------
def load_dataset(file_path: str, text_column: str, label_column: str):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    df = df.dropna(subset=[text_column, label_column])
    df[label_column] = df[label_column].astype(int)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[label_column])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[label_column])
    logger.info(f"Divisão - Treino: {len(train_df)}, Validação: {len(val_df)}, Teste: {len(test_df)}")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df), Dataset.from_pandas(test_df)

def preprocess_data(train_ds, val_ds, test_ds, tokenizer, text_column, label_column, max_length=128):
    def tokenize(batch):
        return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=max_length)
    
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    for split_name, ds in zip(["train", "val", "test"], [train_ds, val_ds, test_ds]):
        ds = ds.map(lambda batch: {"labels": batch[label_column]}, batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        if split_name == "train": train_ds = ds
        elif split_name == "val": val_ds = ds
        else: test_ds = ds
    return train_ds, val_ds, test_ds

# -------------------
# Métricas
# -------------------
def compute_metrics(pred):
    labels = pred.label_ids
    logits = torch.tensor(pred.predictions)
    probs = torch.softmax(logits / TEMPERATURE, dim=1).numpy()
    
    # Aplica threshold apenas na classe 1
    preds = np.argmax(probs, axis=1)
    for i in range(len(probs)):
        if probs[i, 1] >= CLASS_THRESHOLD[1]:
            preds[i] = 1

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', labels=CLASSES, zero_division=0
    )
    metrics = {'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro}

    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0, labels=CLASSES
    )
    for i, (p, r, f1) in enumerate(zip(precision_class, recall_class, f1_class)):
        metrics[f'precision_class_{i}'] = p
        metrics[f'recall_class_{i}'] = r
        metrics[f'f1_class_{i}'] = f1
    return metrics

# -------------------
# Treinamento
# -------------------
def train_model(model_name, train_ds, val_ds, output_dir, epochs, batch_size, lr):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name, num_labels=len(CLASSES))
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)

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
        metric_for_best_model="f1_macro",
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
    metrics = trainer.evaluate(val_ds)

    # Salva modelo e resultados
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "results.csv"), index=False)

    logger.info(f"Treinamento finalizado. Resultados salvos em {os.path.join(output_dir, 'results.csv')}")
    return metrics

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Treinamento BERTimbau multiclass com métricas completas")
    parser.add_argument('--data_path', type=str, default='Treinamento/LI/LI_processado.csv')
    parser.add_argument('--text_column', type=str, default='Frase')
    parser.add_argument('--label_column', type=str, default='Intensidade')
    parser.add_argument('--output_dir', type=str, default='./models/LI')
    parser.add_argument('--model_name', type=str, default='neuralmind/bert-base-portuguese-cased')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    train_ds, val_ds, test_ds = load_dataset(args.data_path, args.text_column, args.label_column)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    train_ds, val_ds, test_ds = preprocess_data(
        train_ds, val_ds, test_ds, tokenizer, args.text_column, args.label_column, args.max_length
    )

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = train_model(
        args.model_name, train_ds, val_ds, args.output_dir,
        args.epochs, args.batch_size, args.learning_rate
    )
    logger.info(f"Métricas finais: {metrics}")

if __name__ == "__main__":
    main()
