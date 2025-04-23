from typing import List
from embd import model, setup_chroma, COLLECTION_NAME

TOP_K = 5


def get_chroma_collection():
    client = setup_chroma()
    return client.get_collection(name=COLLECTION_NAME)


def retrieve_relevant_chunks(query: str, k: int = TOP_K) -> List[str]:
    collection = get_chroma_collection()
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0] 

