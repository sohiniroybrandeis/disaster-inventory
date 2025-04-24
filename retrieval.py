from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index for searching
index = faiss.read_index("disaster_faiss.index")

# Reload the chunk texts (title + content)
df = pd.read_csv("indexed_chunks.csv")
texts = df["chunk"]
texts = texts.tolist()

# Load Gemma 2B Instruct model for generation
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
causal_model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up QA pipeline
qa = pipeline(
    "text-generation", 
    model=causal_model, 
    tokenizer=tokenizer,
    return_full_text=False
)

def answer_question(question, top_k=5):
    print(f"\n=== Question: {question} ===")

    # Retrieve top-k chunks
    query_embedding = embedding_model.encode([question]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [texts[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    # Display context preview
    print("\n[Context Preview]")
    print(context[:500] + "...\n")

    # Construct the prompt
    prompt = (
    "Use the following context to answer the question. "
    "If the answer isn’t clearly stated, try your best to infer it, but don't guess.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n\n"
    "Answer:"
    )


    # Generate and return the answer
    result = qa(prompt, max_new_tokens=200, do_sample=False, temperature=0.1)[0]["generated_text"]
    print("\n[Answer]")
    print(result)

answer_question("Summarize the disasters that occurred in Africa in 2017.")