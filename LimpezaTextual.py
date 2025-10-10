import pandas as pd


df = pd.read_csv("ArquivoCSV", sep=";", encoding="utf-8")


def clean_text(text):
    text = str(text)  
    text = text.replace('"', '')   
    text = text.replace("'", "")   
    text = text.replace(",", "")  
    text = text.strip()            
    return text


df["FRASE"] = df["FRASE"].apply(clean_text)


df.to_csv("dataset_filtrado_limpo.csv", sep=";", index=False, encoding="utf-8")

print("Dataset limpo e salvo como 'dataset_filtrado_limpo.csv'")
