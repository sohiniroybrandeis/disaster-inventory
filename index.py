import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your CSV
df = pd.read_csv("cleaned_rag_dataset.csv")

# Extract the text column
chunks = df["content"].tolist()  # or any column with the main text

# Load a lightweight, accurate embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + good quality

# Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Optional: Save index for later use
faiss.write_index(index, "disaster_faiss.index")
