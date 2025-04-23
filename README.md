### Biomedical RAG Chatbot 

An AI-powered chatbot that answers biomedical questions with text + diagram/image generation using Retrieval-Augmented Generation (RAG).
---

## Features

1. Domain-specific chunking of PubMed PDFs 
    - Implemented Character based-chunking.

2. PubMedBERT embeddings stored in ChromaDB - 
    - Used PubMedBERT for embeddings for better context
    - Used ChromaDB, instead of FAISS as ChromaDB provides more functionality (search with metadata) and works better according to the current use-case.

3. Local semantic search

4. Answer generation using "BioGPT"
    - BioGPT used for answering medical queries in a more relevant manner

5. Text-to-image generation using "Stable Diffusion"

6. Optional image editing
    - Option to edit the output image with the use of prompts by the user

7. Gradio-based frontend
######

######

## HOW TO RUN (Locally)
Run Locally (via Docker)

1. Clone the repo
2. Build Docker image - "docker build -t biomedical-rag-chatbot ."
3. Run the container - "docker run -p 7860:7860 biomedical-rag-chatbot"


## Index Your Biomedical Dataset (One-Time Setup)
- python setup_index.py




