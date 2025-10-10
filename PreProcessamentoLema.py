import pandas as pd
import re
import spacy

nlp = spacy.load("pt_core_news_sm")

df = pd.read_csv("ToxData.csv", sep=";")

def preprocess_lemma(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zá-ú\s]", "", text)
    doc = nlp(text)

    tokens = [token.lemma_ for token in doc 
              if not token.is_stop]
    return " ".join(tokens)

df["FRASE"] = df["FRASE"].apply(preprocess_lemma)

df.to_csv("TOX_processado.csv", index=False, encoding="utf-8")


