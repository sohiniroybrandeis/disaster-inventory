import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt_tab")
from transformers import AutoTokenizer

df = pd.read_csv("scraped_articles.csv") #loading raw data

df.drop_duplicates(subset=["content"], inplace=True) #no duplicates

df.dropna(subset=["content"], inplace=True) #no empties

df["content"] = df["content"].str.strip()
df["title"] = df["title"].str.strip() #no extra whitespace


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text.strip()

df["content"] = df["content"].apply(clean_text)
df["title"] = df["title"].apply(clean_text)

def chunk_text(text, max_length=500):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences: #chunking to a max of 300 characters
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    chunks.append(current_chunk.strip())
    return chunks


df["chunks"] = df["content"].apply(chunk_text)


df = df.explode("chunks", ignore_index=True)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

df["tokens"] = df["chunks"].apply(lambda x: tokenizer.tokenize(x))
df["num_tokens"] = df["tokens"].apply(len)

print(df[["title", "num_tokens"]].head())

df.to_csv("cleaned_rag_dataset.csv", index=False)
print("Cleaned dataset saved!")


