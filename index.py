import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

df = pd.read_csv("cleaned_rag_dataset.csv") #load csv

df["combined"] = df["title"].fillna("") + " - " + df["content"].fillna("") #append title to content

chunks = df["combined"].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2') #embedding model

embeddings = model.encode(chunks, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1]) #index creation
index.add(embeddings)

faiss.write_index(index, "disaster_faiss.index")
df[["combined"]].rename(columns={"combined": "chunk"}).to_csv("indexed_chunks.csv", index=False) #save index and text chunks for retrieval