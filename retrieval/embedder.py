import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIM = 384 # Vector size of model 

# Path to save data into disk
BASE_DIR = Path(__file__).parent
FAISS_INDEX_PATH = BASE_DIR / "faiss_index.index"
BM25_PATH = BASE_DIR / "bm25_index.pkl"
METADATA_PATH = BASE_DIR / "metadata_store.pkl"
PARENT_STORE_PATH = BASE_DIR / "parent_store.pkl"



def embed_chunks(vector_database: list[dict]) -> tuple[list[dict], np.ndarray]:
    """Fill the 'vector' field for each entry in vector_database."""
    # Get text and encode text
    texts = [entry["text"] for entry in vector_database]
    embeddings = _model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True #Normalize for FAISS index (Optimize when searching vector db)
    )
    
    for entry, vector in zip(vector_database, embeddings):
        entry["vector"] = vector.tolist()

    # Save metadata
    metadata_to_save = []
    for entry in vector_database:
        metadata_entry = {
            "child_id": entry["child_id"],
            "text": entry["text"],
            "parent_id": entry["parent_id"],
        }
        metadata_to_save.append(metadata_entry)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_to_save, f)

    return vector_database

def build_faiss_index(vector_embedding):
    # Persistence
    index = faiss.IndexFlatIP(VECTOR_DIM)
    embeddings_np = np.array(vector_embedding).astype("float32")
    index.add(embeddings_np)
    
    # Save faiss index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
def build_bm25_index(metadata_store: list[dict]):
    corpus = [
        item["text"]
        for item in metadata_store
    ]

    tokenized_corpus = [
        doc.lower().split()
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25,f)

def save_parent_store(parent_store: dict):
    with open(PARENT_STORE_PATH, "wb") as f:
        pickle.dump(parent_store, f)
        
def load_parent_store():
    with open(
        PARENT_STORE_PATH,
        "rb"
    ) as f:
        return pickle.load(f)