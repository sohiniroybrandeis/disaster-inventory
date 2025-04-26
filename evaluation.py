from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from rouge_score import rouge_scorer
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

# Assuming qa_eval.csv has columns: ['query', 'relevant_answer']

# Function to retrieve top-k answers using FAISS
def retrieve_top_k(query_embedding, faiss_index, k):
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

# Evaluate using ROUGE
def evaluate_retrieval(qa_data, faiss_index, k):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for idx, row in qa_data.iterrows():
        query = row['query']
        relevant_answer = row['relevant_answer']
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Retrieve the top-k answers using FAISS
        top_k_indices = retrieve_top_k(query_embedding, faiss_index, k)
        
        # Get the retrieved answers
        retrieved_answers = get_answer_from_index(top_k_indices[0])

        # Take only the best retrieved answer (top-1)
        if retrieved_answers:
            best_answer = retrieved_answers[0]
            scores = scorer.score(relevant_answer, best_answer)

            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    # Aggregate ROUGE metrics
    results = {
        "avg_rouge1": np.mean(rouge1_scores),
        "avg_rouge2": np.mean(rouge2_scores),
        "avg_rougeL": np.mean(rougeL_scores)
    }

    return results

# Call the evaluation function
results = evaluate_retrieval(qa_data, index, k=10)

print(f"ROUGE-1: {results['avg_rouge1']:.4f}")
print(f"ROUGE-2: {results['avg_rouge2']:.4f}")
print(f"ROUGE-L: {results['avg_rougeL']:.4f}")