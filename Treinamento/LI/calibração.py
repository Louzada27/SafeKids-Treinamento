import os
import numpy as np
import torch
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import BertForSequenceClassification, BertTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASSES = ["Nenhuma", "Leve", "Severa"]

# --- Helpers: extrai probabilidades e rótulos do dataset (DataFrame com colunas text_col e label_col)
def get_probs_and_labels(model, tokenizer, df, text_col='Frase', label_col='Intensidade', max_length=128, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for text in tqdm(df[text_col].tolist(), desc="Inferindo probs"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            probs.append(p)
    for lab in df[label_col].astype(str).str.strip().tolist():
        labels.append(CLASSES.index(lab))
    return np.vstack(probs), np.array(labels)

# --- Mostra estatísticas simples por classe (probabilidade máxima e a prob da própria classe)
def summary_prob_stats(probs, labels):
    out = {}
    for i, cls in enumerate(CLASSES):
        idx = labels == i
        if idx.sum() == 0:
            out[cls] = None
            continue
        cls_probs = probs[idx, i]
        out[cls] = {
            "n": int(idx.sum()),
            "mean_prob": float(cls_probs.mean()),
            "median_prob": float(np.median(cls_probs)),
            "pct_above_0.4": float((cls_probs >= 0.4).mean()),
            "pct_above_0.5": float((cls_probs >= 0.5).mean()),
            "pct_above_0.6": float((cls_probs >= 0.6).mean())
        }
    return out

# --- Temperature scaling (simple)
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature

def calibrate_temperature(model, tokenizer, val_df, text_col='Frase', label_col='Intensidade', max_length=128, device=None):
    # obtém logits no validation set (antes do softmax)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    logits_list = []
    labels = []
    with torch.no_grad():
        for text in tqdm(val_df[text_col].tolist(), desc="Coletando logits"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
            logits = model(**inputs).logits.cpu()
            logits_list.append(logits.numpy()[0])
    logits = np.vstack(logits_list)
    labels = val_df[label_col].astype(str).str.strip().map({c:i for i,c in enumerate(CLASSES)}).values

    # Torch tensors
    logits_t = torch.tensor(logits)
    labels_t = torch.tensor(labels, dtype=torch.long)

    temp_model = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)

    nll = torch.nn.CrossEntropyLoss()

    def eval_fn():
        optimizer.zero_grad()
        scaled = temp_model(logits_t.to(device))
        loss = nll(scaled, labels_t.to(device))
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    temperature = float(temp_model.temperature.detach().cpu().numpy()[0])
    logger.info(f"Temperatura calibrada: {temperature:.4f}")
    return temperature

# --- Aplica temperatura e procura thresholds por classe (lista thresholds)
def evaluate_thresholds_from_probs(probs, labels, thresholds=np.arange(0.1, 0.91, 0.05)):
    # Retorna por-classe métricas para cada threshold (para prob da própria classe)
    results = {cls: [] for cls in CLASSES}
    for i, cls in enumerate(CLASSES):
        gold = (labels == i).astype(int)
        for t in thresholds:
            preds = (probs[:, i] >= t).astype(int)
            prec, rec, f1, sup = precision_recall_fscore_support(gold, preds, average='binary', zero_division=0)
            results[cls].append({"threshold": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(gold.sum())})
    return results

# --- Avalia predição com thresholds por classe (seleção entre classes que passam seus thresholds)
def predict_with_class_thresholds_from_probs(probs, thresholds_map, default_class='Nenhuma'):
    preds = []
    for p in probs:
        passed = [i for i,cls in enumerate(CLASSES) if p[i] >= thresholds_map[cls]]
        if len(passed) == 0:
            preds.append(CLASSES.index(default_class))
        else:
            # escolhe a que tiver maior prob entre as que passaram
            best = max(passed, key=lambda j: p[j])
            preds.append(best)
    return np.array(preds)

# --- Util: calcula relatório
def report_from_preds(gold_labels, pred_labels):
    print(classification_report(gold_labels, pred_labels, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(gold_labels, pred_labels)
    print("Confusion matrix:\n", cm)

# -----------------------
# Exemplo de uso (main)
# -----------------------
if __name__ == "__main__":
    # Paths: ajuste se necessário
    MODEL_DIR = "./models/bertimbau_multiclass"
    VAL_PATH = "Treinamento/LI/LI_processado.csv"   # você pode separar um val/test
    TEXT_COL = "Frase"
    LABEL_COL = "Intensidade"

    logger.info("Carregando modelo/tokenizer...")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-base-portuguese-cased")

    # Carregue seu conjunto de validação (ou use test set)
    df = pd.read_csv(VAL_PATH).dropna(subset=[TEXT_COL, LABEL_COL])
    # opcional: se já tem split, carregue apenas val/test
    # Aqui uso todo df como exemplo
    probs, labels = get_probs_and_labels(model, tokenizer, df, TEXT_COL, LABEL_COL)

    # 1) estatísticas por classe
    stats = summary_prob_stats(probs, labels)
    logger.info("Stats por classe (prob da própria classe):")
    for k, v in stats.items():
        logger.info(f"{k}: {v}")

    # 2) avaliar thresholds sem calibrar
    thresholds = np.arange(0.1, 0.91, 0.05)
    raw_results = evaluate_thresholds_from_probs(probs, labels, thresholds)
    # Salva para inspeção
    import json
    with open("thresholds_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved thresholds_raw_results.json")

    # 3) calibrar temperatura e reavaliar
    try:
        temp = calibrate_temperature(model, tokenizer, df, TEXT_COL, LABEL_COL)
        scaled_logits = np.log(probs + 1e-12) / temp
        scaled_probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
        cal_results = evaluate_thresholds_from_probs(scaled_probs, labels, thresholds)
        with open("thresholds_calibrated_results.json", "w", encoding="utf-8") as f:
            json.dump(cal_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved thresholds_calibrated_results.json")
    except Exception as e:
        logger.warning(f"Falha na calibração por temperatura: {e}")

    # 4) exemplo: aplicar thresholds por classe (use os seus melhores)
    # Exemplo inicial (ajuste conforme seus resultados)
    thresh_map = {"Nenhuma": 0.7, "Leve": 0.35, "Severa": 0.6}
    preds = predict_with_class_thresholds_from_probs(probs, thresh_map, default_class='Nenhuma')
    report_from_preds(labels, preds)
