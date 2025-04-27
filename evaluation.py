from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reload the chunk texts (title + content)
df = pd.read_csv("indexed_chunks.csv")
texts = df["chunk"]
texts = texts.tolist()

# Load FAISS index from the file
index = faiss.read_index("disaster_faiss.index")

# Load your evaluation CSV (qa_eval.csv)
qa_data = pd.read_csv('qa_eval.csv', quotechar='"')

# Function to retrieve top-k answers using FAISS
def retrieve_top_k(query_embedding, faiss_index, k):
    D, I = faiss_index.search(query_embedding, k)
    return I

# Define get_embedding
def get_embedding(query):
    return embedding_model.encode([query]).astype("float32")

# Define get_answer_from_index
def get_answer_from_index(indices):
    return [texts[i] for i in indices]

# Evaluate using BLEU
def evaluate_retrieval_bleu(qa_data, faiss_index, k=10):
    bleu_scores = []
    smoothing = SmoothingFunction().method4  # Smoothing to avoid zero scores

    for idx, row in qa_data.iterrows():
        query = row['query']
        relevant_answer = row['relevant_answer']
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Retrieve top-k answers
        top_k_indices = retrieve_top_k(query_embedding, faiss_index, k)
        
        # Get the retrieved answers
        retrieved_answers = get_answer_from_index(top_k_indices[0])

        # Debugging: Print retrieved answers for each query
        print(f"Query: {query}")
        print("Top-k Retrieved Answers:")
        for i, ans in enumerate(retrieved_answers):
            print(f"{i+1}: {ans}")

        if retrieved_answers:
            best_answer = retrieved_answers[0]
            reference = relevant_answer.split()  # Tokenized reference
            hypothesis = best_answer.split()      # Tokenized generation

            bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=smoothing)
            bleu_scores.append(bleu_score)

    results = {
        "avg_bleu": np.mean(bleu_scores)
    }

    return results

# Call the BLEU evaluation function
results = evaluate_retrieval_bleu(qa_data, index, k=10)

print(f"BLEU Score: {results['avg_bleu']:.4f}")