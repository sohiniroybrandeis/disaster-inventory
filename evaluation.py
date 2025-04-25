from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reload the chunk texts (title + content)
df = pd.read_csv("indexed_chunks.csv")
texts = df["chunk"]
texts = texts.tolist()

# Load FAISS index from the file
index = faiss.read_index("disaster_faiss.index")

# Load your evaluation CSV (qa_eval.csv)
qa_data = pd.read_csv('qa_eval.csv')

# Assuming qa_eval.csv has columns: ['query', 'relevant_answer']

# Function to retrieve top-k answers using FAISS
def retrieve_top_k(query_embedding, faiss_index, k=5):
    # Perform the retrieval (this assumes query_embedding and faiss_index are available)
    # Querying the FAISS index
    D, I = faiss_index.search(query_embedding, k)  # D = distances, I = indices
    return I

# Define get_embedding using your SentenceTransformer model
def get_embedding(query):
    # Use the preloaded SentenceTransformer model to get embeddings
    return embedding_model.encode([query]).astype("float32")

# Define get_answer_from_index to retrieve relevant answers based on FAISS indices
def get_answer_from_index(indices):
    # Get the actual answers based on the indices
    return [texts[i] for i in indices]

# Evaluate Precision, Recall, F1-Score, MRR
def evaluate_retrieval(qa_data, faiss_index, k=5):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    mrr_scores = []

    for idx, row in qa_data.iterrows():
        query = row['query']
        relevant_answer = row['relevant_answer']
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Retrieve the top-k answers using FAISS
        top_k_indices = retrieve_top_k(query_embedding, faiss_index, k)
        
        # Get the retrieved answers based on indices
        retrieved_answers = get_answer_from_index(top_k_indices[0])
        
        # Check if the relevant answer is in the top-k retrieved answers
        is_relevant = [1 if relevant_answer == ans else 0 for ans in retrieved_answers]
        
        # Calculate Precision, Recall, F1, and MRR for this query
        precision = precision_score([1] * len(is_relevant), is_relevant, average='binary', zero_division=0)
        recall = recall_score([1] * len(is_relevant), is_relevant, average='binary', zero_division=0)
        f1 = f1_score([1] * len(is_relevant), is_relevant, average='binary', zero_division=0)
        
        # MRR calculation: find first relevant answer in the top-k
        mrr = 0
        if 1 in is_relevant:
            mrr = 1 / (is_relevant.index(1) + 1)  # MRR for the first relevant result

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        mrr_scores.append(mrr)
    
    # Calculate average metrics
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_mrr = np.mean(mrr_scores)

    return avg_precision, avg_recall, avg_f1, avg_mrr

# Call the evaluation function
avg_precision, avg_recall, avg_f1, avg_mrr = evaluate_retrieval(qa_data, index, k=5)

print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1-Score: {avg_f1:.4f}")
print(f"MRR: {avg_mrr:.4f}")