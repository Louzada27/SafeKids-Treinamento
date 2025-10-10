import pandas as pd
import re
import spacy
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
import nltk


nltk.download("stopwords")

nlp = spacy.load("pt_core_news_sm")
stemmer = RSLPStemmer()
stop_words = set(stopwords.words("portuguese"))

df = pd.read_csv("LiData.csv",sep=";")

def preprocess_stem(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zá-ú\s]", "", text)
    doc = nlp(text)
   
    tokens = [stemmer.stem(token.text) for token in doc 
              if token.text not in stop_words]
    return " ".join(tokens)

df["Frase"] = df["Frase"].apply(preprocess_stem)

df.to_csv("LI_processado.csv", index=False, encoding="utf-8")
