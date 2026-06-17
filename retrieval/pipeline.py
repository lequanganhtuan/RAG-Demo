from ingestion.pipeline import run_pipeline
from pathlib import Path
from retrieval.embedder import (
    embed_chunks,
    build_faiss_index,
    save_parent_store
)

def build_retrieval_index(document_path):
    vector_database, parent_store = run_pipeline(document_path)
    vector_database = embed_chunks(vector_database)

    embeddings = [
        entry["vector"]
        for entry in vector_database
    ]

    build_faiss_index(embeddings)
    save_parent_store(parent_store)

    return parent_store

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    store = build_retrieval_index(BASE_DIR / "RAG_test.pdf")
    print(f"Finish module 2")