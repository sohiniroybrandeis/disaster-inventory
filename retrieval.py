from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import dateparser
from datetime import datetime
from word2number import w2n


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

def convert_words_to_numbers(text):
    try:
        words = text.split()
        converted = []
        for i, word in enumerate(words):
            try:
                num = w2n.word_to_num(word)
                converted.append(str(num))
            except:
                converted.append(word)
        return ' '.join(converted)
    except:
        return text
    
def extract_year_from_question(question):
    question = convert_words_to_numbers(question)  # <-- important!
    years = set()

    # Extract explicit years
    match = re.findall(r"\b(201[5-9]|202[0-5])\b", question)
    years.update(map(int, match))

    # Handle relative range
    range_match = re.search(r"over the last (\d+) years", question, re.IGNORECASE)
    if range_match:
        n_years = int(range_match.group(1))
        current_year = datetime.now().year
        for y in range(current_year - n_years + 1, current_year + 1):
            if 2015 <= y <= 2025:
                years.add(y)

    return sorted(list(years))


def answer_question(question, top_k=5):
    print(f"\n=== Question: {question} ===")

    year = extract_year_from_question(question)
    query_embedding = embedding_model.encode([question]).astype("float32")

    if year:
        # Filter texts that contain the target year
        filtered_texts = [t for t in texts if str(year) in t]
        
        if filtered_texts:
            # Create temporary FAISS index on the filtered texts
            filtered_embeddings = embedding_model.encode(filtered_texts).astype("float32")
            temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
            temp_index.add(filtered_embeddings)
            distances, indices = temp_index.search(query_embedding, top_k)
            retrieved_chunks = [filtered_texts[i] for i in indices[0]]
        else:
            print(f"\n[Notice] No matches found for year {year}. Falling back to full search.")
            distances, indices = index.search(query_embedding, top_k)
            retrieved_chunks = [texts[i] for i in indices[0]]
    else:
        # No year provided — use full index
        distances, indices = index.search(query_embedding, top_k)
        retrieved_chunks = [texts[i] for i in indices[0]]

    context = "\n".join(retrieved_chunks)

    # Display context preview
    print("\n[Context Preview]")
    print(context[:500] + "...\n")

    # Prompt construction
    prompt = (
        "Use the following context to answer the question. "
        "If the answer isn’t clearly stated, try your best to infer it, but don't guess.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Generate answer
    result = qa(prompt, max_new_tokens=200, do_sample=False, temperature=0.1)[0]["generated_text"]
    
    print("\n[Answer]")
    print(result)

answer_question("Summarize the disasters that occurred in Africa over the last two years.")