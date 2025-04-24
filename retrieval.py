from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
import re
import torch

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index for searching
index = faiss.read_index("disaster_faiss.index")

# Reload the original texts
df = pd.read_csv("cleaned_rag_dataset.csv")
texts = df["content"].tolist()

# Load Meta LLaMA 3 Instruct model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
causal_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set up the text-generation pipeline
qa = pipeline("text-generation", model=causal_model, tokenizer=tokenizer)

def truncate_text(context: str, question: str, tokenizer: PreTrainedTokenizerBase, max_tokens=4096, reserved_for_prompt=200) -> str:
    max_context_tokens = max_tokens - reserved_for_prompt
    inputs = tokenizer(context, max_length=max_context_tokens, truncation=True, return_tensors="pt")
    return tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

def extract_year(question):
    match = re.search(r"\b(20[0-2][0-9])\b", question)
    return match.group(1) if match else None

def filter_by_year(chunks, year):
    return [chunk for chunk in chunks if year in chunk]

def answer_question(question, top_k=2):
    # Encode and search
    question_embedding = embedding_model.encode([question]).astype("float32")
    _, I = index.search(question_embedding, top_k)
    retrieved_chunks = [texts[i] for i in I[0]]

    # Optional: Filter by year if found
    year = extract_year(question)
    if year:
        filtered_chunks = filter_by_year(retrieved_chunks, year)
        if filtered_chunks:
            retrieved_chunks = filtered_chunks

    # Prepare context and prompt
    context = "\n".join(retrieved_chunks)
    context = truncate_text(context, question, tokenizer)
    
    prompt = f"<|begin_of_text|><|user|>\nContext:\n{context}\n\nQuestion: {question}<|end_of_text|>\n<|assistant|>"

    # Generate response
    response_text = qa(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
    
    # Strip prompt echoes if needed
    response = response_text.split("<|assistant|>")[-1].strip()

    print("\nAnswer:", response)

# Example usage
answer_question("Give me a summary of the disasters that occurred in 2019.")