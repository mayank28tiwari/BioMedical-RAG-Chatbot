import os
import pickle
from typing import List
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "neuml/pubmedbert-base-embeddings"
PERSIST_DIR = "chroma_storage"
COLLECTION_NAME = "biomed_chunks"
PKL_FILE = "all_chunks.pkl"

model = SentenceTransformer(MODEL_NAME)

def setup_chroma():
    client = chromadb.Client()
    return client

def populate_chroma(pkl_file: str = PKL_FILE) -> None:
    with open(pkl_file, "rb") as f:
        all_chunks, metadata = pickle.load(f)

    ids = [f"chunk-{i}" for i in range(len(all_chunks))]
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    client = setup_chroma()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )

    return collection