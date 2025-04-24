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
    
    # Tokenize context into sentences or chunks
    sentences = context.split('\n')  # Split by newlines, or you can use a sentence tokenizer if available
    
    # Encode the context to get the number of tokens
    context_tokens = tokenizer.encode(context, truncation=False, return_tensors="pt")[0]
    
    # Check if the context exceeds the max token limit
    if len(context_tokens) > max_context_tokens:
        print(f"Truncating context to {max_context_tokens} tokens...")
        
        # Keep the sentences until the token limit is reached
        truncated_context = ''
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, truncation=False, return_tensors="pt")[0]
            if current_tokens + len(sentence_tokens) <= max_context_tokens:
                truncated_context += sentence + '\n'
                current_tokens += len(sentence_tokens)
            else:
                break
        
        context = truncated_context.strip()  # Remove extra newline at the end
    
    return context


# Function to answer a question based on context retrieved via FAISS
def answer_question(question, top_k=2):
    # Step 1: Encode the question and perform search in FAISS
    question_embedding = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_embedding, top_k)
    
    # Step 2: Retrieve relevant context from the dataset
    retrieved_chunks = [texts[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)
    context = truncate_text(context, question, max_tokens=512, reserved_for_prompt=100)
    
    # Step 3: Prepare the prompt with the retrieved context and the question
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}"
    
    # Step 4: Generate the answer using the causal language model (Gemma)
    response = qa(prompt, max_new_tokens=200)[0]['generated_text']  # Using max_new_tokens instead of max_length
    
    print("\nAnswer:", response)

# Example usage
answer_question("Summarize disasters that happened in 2025 in Africa.")