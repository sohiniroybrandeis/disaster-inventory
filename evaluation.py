from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import bert_score

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("indexed_chunks.csv") #reload the chunk texts (title + content)
texts = df["chunk"]
texts = texts.tolist()

index = faiss.read_index("disaster_faiss.index") #load FAISS index

qa_data = pd.read_csv('qa_eval.csv', quotechar='"') #load evaluation CSV (qa_eval.csv)

def retrieve_top_k(query_embedding, faiss_index, k): #retrieve top-k answers using embeddings
    D, I = faiss_index.search(query_embedding, k)
    return I

def get_embedding(query):
    return embedding_model.encode([query]).astype("float32")

def get_answer_from_index(indices):
    return [texts[i] for i in indices]

def evaluate_retrieval_bertscore(qa_data, faiss_index, k=10): # evaluate using BERTScore
    references = []
    candidates = []

    for idx, row in qa_data.iterrows():
        query = row['query']
        relevant_answer = row['relevant_answer']
        
        query_embedding = get_embedding(query)
        top_k_indices = retrieve_top_k(query_embedding, faiss_index, k)
        retrieved_answers = get_answer_from_index(top_k_indices[0])

        # for i, ans in enumerate(retrieved_answers):
        #     print(f"{i+1}: {ans}")

        if retrieved_answers:
            best_answer = retrieved_answers[0]
            candidates.append(best_answer)
            references.append(relevant_answer)

    P, R, F1 = bert_score.score(candidates, references, lang="en", verbose=True) #compute score
    
    results = {
        "avg_precision": P.mean().item(),
        "avg_recall": R.mean().item(),
        "avg_f1": F1.mean().item()
    }

    return results


results = evaluate_retrieval_bertscore(qa_data, index, k=10)

print(f"BERTScore Precision: {results['avg_precision']:.4f}")
print(f"BERTScore Recall: {results['avg_recall']:.4f}")
print(f"BERTScore F1: {results['avg_f1']:.4f}")