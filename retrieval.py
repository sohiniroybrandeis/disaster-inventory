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
texts = df["chunk"].tolist()

# Load Gemma 2B Instruct model for text generation
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
causal_model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up the text-generation pipeline for Gemma with deterministic output
qa = pipeline(
    "text-generation", 
    model=causal_model, 
    tokenizer=tokenizer,
    return_full_text=False
)

# Function to truncate context text to fit within the token limit
def truncate_text(context, question, max_tokens=512, reserved_for_prompt=100):
    max_context_tokens = max_tokens - reserved_for_prompt
    sentences = context.split('. ')
    
    truncated_context = ''
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence += '. '
        sentence_tokens = tokenizer.encode(sentence, truncation=False, return_tensors="pt")[0]
        if current_tokens + len(sentence_tokens) <= max_context_tokens:
            truncated_context += sentence
            current_tokens += len(sentence_tokens)
        else:
            break

    print("Original context tokens:", len(tokenizer.encode(context)))
    print("Truncated to tokens:", current_tokens)
    return truncated_context.strip()

# Function to answer a question based on context retrieved via FAISS
def answer_question(question, top_k=2):
    print(f"\n=== Question: {question} ===\n")

    # Step 1: Encode the question and perform search in FAISS
    question_embedding = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_embedding, top_k)

    # Step 2: Retrieve relevant context from the dataset
    retrieved_chunks = [texts[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)
    context = truncate_text(context, question, max_tokens=512, reserved_for_prompt=100)

    # Print a preview of the context
    print("\n[Context Preview]")
    print(context[:500], "\n...")

    # Step 3: Prepare the prompt with improved clarity
    prompt = (
        f"Answer the following question using only the information from the context below.\n"
        f"If the answer is not available, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Step 4: Generate the answer using the causal language model
    response = qa(prompt, max_new_tokens=200, do_sample=False, temperature=0.1)[0]['generated_text']

    print("\n[Raw Model Output]")
    print(response)

# Example usage
answer_question("Summarize disasters that happened in 2025 in Africa.")