### Task

This is a Retrieval Augmented Generation (RAG) system relating to environmental sciences/data, more specifically focusing on information retrieval about natural disasters and environmental/humanitarian conflicts occurring in African countries from 2015-2025. Some sample queries that can be asked are:

Question: What were the primary damages caused by the flooding in Guinea in August 2021?
Answer: The flooding in Guinea resulted in the destruction of 763 water points, 103 host families, 21 resettlement sites, 964 latrines, and significant impacts on agriculture and animal husbandry.

Question: What were the impacts of Tropical Cyclone 'Luban' on Yemen as of 17 October 2018?
Answer: Tropical Cyclone 'Luban' caused the displacement of more than 3,000 households in Yemen, with significant flooding and damage to homes. Three deaths, 14 missing persons, and over 100 injuries were reported. The cyclone was downgraded to a Tropical Depression by 15 October.

With more up-to-date information on recent events, people/governments in disaster hotspots can stay prepared/take action to improve response times and resource allocation. Identifying hotspots of frequent disasters can help prioritize investments in infrastructure, early warning systems, and community resilience programs.


### Code Structure

Code is organized into the given files:

- relief_scraping.py
  - All code to scrape data off https://reliefweb.int/ and save to a CSV.
- relief_preprocessing.py
  - All code to preprocess data by dropping duplicate and empty rows from the CSV, and clean and chunk text.
- index.py
  - All code to chunk the dataset, generate embeddings using a pre-trained Sentence Transformer model, and index them with FAISS for efficient similarity search. It saves the FAISS index and indexed chunks for later retrieval.
- retrieval.py
  - All code to load pre-trained models for embedding generation and text summarization, extract relevant years from a question, search disaster data using FAISS, and generate a summary based on the retrieved context. It then outputs a concise summary answering the user's question.
- evaluation.py
  - All code to retrieve top-k answers to a sample question from FAISS, and evaluate the system using BertSCORE.

### Running Code

After cloning the github repository, all requirements can be installed by running pip install -r requirements.txt. Make sure to be in the same directory as the requirements file. Retrieval can be done by running retrieval.py, and evaluation can be done by running evaluation.py.
