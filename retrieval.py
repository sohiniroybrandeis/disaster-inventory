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

# Load LLaMA 3B/8B Instruct model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # or 3B if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
causal_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Will place model on GPU if available
)

# Set up the text-generation pipeline
qa = pipeline("text-generation", model=causal_model, tokenizer=tokenizer)

def truncate_text(context, question, max_tokens=4096, reserved_for_prompt=200):
    max_context_tokens = max_tokens - reserved_for_prompt
    tokens = tokenizer.encode(context, truncation=True, max_length=max_context_tokens, return_tensors="pt")[0]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def extract_year(question):
    match = re.search(r"\b(20[0-2][0-9])\b", question)
    return match.group(1) if match else None

def filter_by_year(chunks, year):
    return [chunk for chunk in chunks if year in chunk]

def answer_question(question, top_k=2):
    question_embedding = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_embedding, top_k)

    retrieved_chunks = [texts[i] for i in I[0]]
    year = extract_year(question)
    if year:
        filtered_chunks = filter_by_year(retrieved_chunks, year)
        if filtered_chunks:
            retrieved_chunks = filtered_chunks

    context = "\n".join(retrieved_chunks)
    context = truncate_text(context, question)

    # Prompt format tailored for LLaMA 3 Instruct
    prompt = f"<|begin_of_text|><|user|>\nAnswer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}<|end_of_text|>\n<|assistant|>"

    response = qa(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
    print("\nAnswer:", response.split("<|assistant|>")[-1].strip())

# Example usage
answer_question("Give me a summary of the disasters that occurred in 2019.")