#!/usr/bin/env python3
"""
Avaliação de modelo BERTimbau multiclass (0, 1, 2) com variação de threshold
para a classe 1, buscando o melhor F1-score.

O script:
- Carrega modelo e tokenizer treinados
- Carrega dataset de teste
- Calcula métricas (Precision, Recall, F1) variando o threshold da classe 1
- Exibe e salva o melhor threshold encontrado
"""

import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertTokenizerFast

# ---------------------
# CONFIGURAÇÕES
# ---------------------
MODEL_DIR = " "  # pasta onde o modelo treinado foi salvo
DATA_PATH = "Treinamento/LI/LI_processado.csv"
TEXT_COLUMN = "Frase"
LABEL_COLUMN = "Intensidade"
THRESHOLDS = np.arange(0.1, 0.91, 0.05)  
CLASSES = [0, 1, 2]
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------
# FUNÇÕES AUXILIARES
# ---------------------
def load_dataset(file_path, text_column, label_column):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    df = df.dropna(subset=[text_column, label_column])
    df[label_column] = df[label_column].astype(int)
    return Dataset.from_pandas(df)

def preprocess_data(ds, tokenizer, text_column, label_column, max_length=128):
    def tokenize(batch):
        return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=max_length)
    ds = ds.map(tokenize, batched=True)
    ds = ds.map(lambda batch: {"labels": batch[label_column]}, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

def evaluate_thresholds(model, ds, thresholds):
    all_metrics = []
    model.eval()

    # gera logits para todos os exemplos
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(ds, batch_size=16):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # testando thresholds para classe 1
    for t in thresholds:
        preds = np.argmax(all_probs, axis=1).copy()
        class1_mask = all_probs[:, 1] >= t
        preds[class1_mask] = 1  # força classe 1 quando probabilidade >= threshold

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average=None, labels=CLASSES, zero_division=0
        )
        all_metrics.append({
            "threshold": t,
            "precision_class_1": precision[1],
            "recall_class_1": recall[1],
            "f1_class_1": f1[1]
        })

    return pd.DataFrame(all_metrics)

# ---------------------
# MAIN
# ---------------------
def main():
    print("Carregando modelo e tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    print("Carregando dataset de teste...")
    ds = load_dataset(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)
    ds = preprocess_data(ds, tokenizer, TEXT_COLUMN, LABEL_COLUMN, MAX_LENGTH)

    print("Avaliando thresholds...")
    results_df = evaluate_thresholds(model, ds, THRESHOLDS)

    best_row = results_df.loc[results_df["f1_class_1"].idxmax()]
    print("\nMelhor threshold encontrado:")
    print(best_row)

    os.makedirs(MODEL_DIR, exist_ok=True)
    results_path = os.path.join(MODEL_DIR, "threshold_evaluation.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados salvos em: {results_path}")

if __name__ == "__main__":
    main()
