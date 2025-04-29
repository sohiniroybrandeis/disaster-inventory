from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
import dateparser
from datetime import datetime
from word2number import w2n
import torch

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
    return_full_text=False,
    device="cpu"  # <- This prevents CUDA memory usage entirely
)

def convert_words_to_numbers(text):
    # Convert spelled-out numbers to digits
    try:
        return re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b',
                      lambda x: str(w2n.word_to_num(x.group())), text, flags=re.IGNORECASE)
    except:
        return text

def extract_relevant_years(question, start_year=2015, end_year=2025):
    current_year = datetime.now().year
    question = convert_words_to_numbers(question)

    years = set()

    # Match explicit years like 2019, 2023
    year_matches = re.findall(r"\b(20[1-2][0-9])\b", question)
    for match in year_matches:
        year = int(match)
        if start_year <= year <= end_year:
            years.add(year)

    # Match relative expressions like "last 2 years"
    rel_match = re.search(r"last (\d{1,2}) years", question, re.IGNORECASE)
    if rel_match:
        n_years = int(rel_match.group(1))
        for i in range(n_years):
            year = current_year - i
            if start_year <= year <= end_year:
                years.add(year)

    return sorted(years)


def answer_question(question, top_k=5):
    print(f"\n=== Question: {question} ===")

    years = extract_relevant_years(question)
    print(f"[Extracted Years] {years if years else 'None'}")

    with torch.no_grad():
        query_embedding = embedding_model.encode([question]).astype("float32")

        if years:
            filtered_texts = [t for t in texts if any(str(y) in t for y in years)]
            if not filtered_texts:
                return "Sorry, I couldn't find any disaster data related to those years."

            filtered_embeddings = embedding_model.encode(filtered_texts).astype("float32")
            temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
            temp_index.add(filtered_embeddings)
            distances, indices = temp_index.search(query_embedding, top_k)
            retrieved_chunks = [filtered_texts[i] for i in indices[0]]
        else:
            distances, indices = index.search(query_embedding, top_k)
            retrieved_chunks = [texts[i] for i in indices[0]]

    context = "\n".join(retrieved_chunks)
    print("\n[Context Preview]")
    print(context[:500] + "...\n")

    prompt = (
        "You are a helpful assistant summarizing recent disaster events in Africa based on the context below. "
        "Use the information to write a concise summary, highlighting locations, dates, and impacts."
        "Include any specific information on disaster relief and how to take action beforehand."
        "If only partial information is available, summarize what is known.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Summary:"
    )


    with torch.no_grad():
        result = qa(prompt, max_new_tokens=300, do_sample=False, temperature=0.1)[0]["generated_text"]
    torch.cuda.empty_cache()

    print("\n[Answer]")
    print(result)

answer_question("What are the most common types of natural disasters in Africa from 2015 to 2025?")