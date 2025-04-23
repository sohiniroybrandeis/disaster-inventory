### Dataset Description

As I mentioned in my proposal, my chosen track for the final project is Track 2: Retrieval-Augmented Generation (RAG). I have chosen to build a system relating to environmental sciences/data, more specifically focusing on information retrieval about natural disasters occurring in African countries from 2015-2025. I have selected this subset of years as a starting point, but can easily add more to it if it is not enough. I have sourced all of my data from https://reliefweb.int/.

For now, I have stored my cleaned data in `cleaned_rag_dataset`. The "raw" data is in `scraped_articles.csv`. I have attempted to chunk the data by splitting my documents into smaller chunks e.g., 300 tokens each.