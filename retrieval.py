from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index for searching
index = faiss.read_index("disaster_faiss.index")

# Reload the original texts (assuming this has disaster-related information)
df = pd.read_csv("cleaned_rag_dataset.csv")
texts = df["content"].tolist()

# Load Gemma 2B Instruct model for text generation
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
causal_model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up the text-generation pipeline for Gemma
qa = pipeline("text-generation", model=causal_model, tokenizer=tokenizer)

# Function to truncate context text to fit within the token limit
def truncate_text(context, question, max_tokens=512, reserved_for_prompt=100):
    # Reserve tokens for the prompt and question
    max_context_tokens = max_tokens - reserved_for_prompt
    tokens = tokenizer.encode(context, truncation=True, max_length=max_context_tokens, return_tensors="pt")[0]
    return tokenizer.decode(tokens, skip_special_tokens=True)

import re

def extract_year(question):
    match = re.search(r"\b(20[0-2][0-9])\b", question)
    return match.group(1) if match else None

def filter_by_year(chunks, year):
    return [chunk for chunk in chunks if year in chunk]

def answer_question(question, top_k=2):
    # Step 1: Encode question and perform FAISS search
    question_embedding = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_embedding, top_k)

    # Step 2: Retrieve initial chunks
    retrieved_chunks = [texts[i] for i in I[0]]

    # Step 3: Filter chunks if a year is detected in the question
    year = extract_year(question)
    if year:
        filtered_chunks = filter_by_year(retrieved_chunks, year)
        if filtered_chunks:
            retrieved_chunks = filtered_chunks

    # Step 4: Truncate context if needed
    context = "\n".join(retrieved_chunks)
    context = truncate_text(context, question, max_tokens=512, reserved_for_prompt=100)

    # Step 5: Build prompt and generate response
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}"
    response = qa(prompt, max_new_tokens=200)[0]['generated_text']
    
    print("\nAnswer:", response)


# Example usage
answer_question("Summary of the disasters that took place in 2017?")
