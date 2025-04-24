from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import torch

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index for searching
index = faiss.read_index("disaster_faiss.index")

# Reload the original texts
df = pd.read_csv("cleaned_rag_dataset.csv")
texts = df["content"].tolist()

# Load LLaMA 3.2 1B Instruct model
model_name = "meta-llama/Meta-Llama-3-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
causal_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

qa = pipeline("text-generation", model=causal_model, tokenizer=tokenizer, return_full_text=False)

# Function to extract year from question
def extract_year(question):
    match = re.search(r"\b(20[0-2][0-9])\b", question)
    return match.group(1) if match else None

# Function to filter retrieved chunks by year
def filter_by_year(chunks, year):
    return [chunk for chunk in chunks if year in chunk]

# Function to truncate context text while reserving space for instructions
def truncate_text(context, max_tokens=4096, reserved_for_prompt=200):
    full_text = f"<|begin_of_text|><|system|>\nYou are a helpful assistant answering questions about natural disasters.\n<|eot_id|>\n<|user|>\n{context}<|eot_id|>\n<|assistant|>\n"
    max_length = max_tokens - reserved_for_prompt
    tokens = tokenizer.encode(full_text, truncation=True, max_length=max_length, return_tensors="pt")[0]
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Function to retrieve and answer questions
def answer_question(question, top_k=2):
    # Step 1: Embed question and perform FAISS search
    question_embedding = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_embedding, top_k)

    # Step 2: Retrieve relevant chunks
    retrieved_chunks = [texts[i] for i in I[0]]

    # Step 3: Apply year filtering if necessary
    year = extract_year(question)
    if year:
        filtered_chunks = filter_by_year(retrieved_chunks, year)
        if filtered_chunks:
            retrieved_chunks = filtered_chunks

    # Step 4: Truncate context while reserving space
    context = "\n".join(retrieved_chunks)
    context = truncate_text(context)

    # Step 5: Format prompt for LLaMA 3
    prompt = (
        "<|begin_of_text|><|system|>\n"
        "You are a helpful assistant answering questions about natural disasters.\n"
        "<|eot_id|>\n<|user|>\n"
        f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}\n"
        "<|eot_id|>\n<|assistant|>\n"
    )

    # Step 6: Generate response
    response = qa(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
    
    print("\nAnswer:", response)

# Example usage
answer_question("Summary of the disasters that took place in 2019?")