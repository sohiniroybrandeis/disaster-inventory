import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your CSV
df = pd.read_csv("cleaned_rag_dataset.csv")

# Combine the title (which contains the year) with the content
df["combined"] = df["title"].fillna("") + " - " + df["content"].fillna("")

# Extract combined chunks
chunks = df["combined"].tolist()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and text chunks for retrieval
faiss.write_index(index, "disaster_faiss.index")
df[["combined"]].rename(columns={"combined": "chunk"}).to_csv("indexed_chunks.csv", index=False)