import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your CSV
df = pd.read_csv("cleaned_rag_dataset.csv")

# Combine title and content for better context
# Use newline for clearer separation
df["chunk"] = df["title"].fillna("") + "\n" + df["content"].fillna("")
chunks = df["chunk"].tolist()

# Load a lightweight, accurate embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + good quality

for i in range(3):  # show a few sample entries
    print(f"\n--- Example chunk {i} ---")
    print(f"Title: {df['title'][i]}")
    print(f"Content: {df['content'][i]}")
    print(f"Full chunk: {df['title'][i]} - {df['content'][i]}")

# Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and text chunks for retrieval
faiss.write_index(index, "disaster_faiss.index")
df[["chunk"]].to_csv("indexed_chunks.csv", index=False)