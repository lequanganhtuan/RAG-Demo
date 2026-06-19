import pickle
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .embedder import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
)

BASE_DIR = FAISS_INDEX_PATH.parent
BM25_PATH = BASE_DIR / "bm25_index.pkl"

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_metadata():
    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)


def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)

def search_faiss(query_text: str,top_k: int = 20) -> list[dict]:
    metadata_store = load_metadata()

    # Load faiss index
    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )
    
    # encode query vector
    query_vector = _model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype("float32")

    #Get the score from result search
    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    
    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        if idx == -1:
            continue
        metadata = metadata_store[idx]
        results.append({
            "score": float(score),
            "child_id": metadata["child_id"],
            "parent_id": metadata["parent_id"],
            "text": metadata["text"]
        })

    return results



def search_bm25(query_text: str,top_k: int = 20) -> list[dict]:
    metadata_store = load_metadata()
    # Load bm25 index
    bm25_index = load_bm25()
    # Format query
    query_tokens = (
        query_text.lower().split()
    )
    # Get the score from result search
    scores = bm25_index.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for idx in ranked_indices:
        metadata = metadata_store[idx]
        results.append({
            "score": float(scores[idx]),
            "child_id": metadata["child_id"],
            "parent_id": metadata["parent_id"],
            "text": metadata["text"]
        })

    return results

def reciprocal_rank_fusion(faiss_results: list[dict],bm25_results: list[dict],k: int = 60
) -> list[dict]:

    fused_scores = {}

    # FAISS rank
    for rank, result in enumerate(
        faiss_results,
        start=1
    ):
        chunk_id = result["child_id"]
        if chunk_id not in fused_scores:
            fused_scores[chunk_id] = {
                "rrf_score": 0.0,
                "data": result
            }
        fused_scores[chunk_id]["rrf_score"] += (
            1 / (k + rank)
        )

    # BM25 rank
    for rank, result in enumerate(
        bm25_results,
        start=1
    ):
        chunk_id = result["child_id"]
        if chunk_id not in fused_scores:
            fused_scores[chunk_id] = {
                "rrf_score": 0.0,
                "data": result
            }
        fused_scores[chunk_id]["rrf_score"] += (
            1 / (k + rank)
        )

    fused_list = []

    # Change format from dict score to list 
    for item in fused_scores.values():
        result = item["data"].copy()
        result["rrf_score"] = (
            item["rrf_score"]
        )
        fused_list.append(result)

    fused_list.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    return fused_list

def search_hybrid(
    query_text: str,
    top_k: int = 20
) -> list[dict]:

    faiss_results = search_faiss(
        query_text=query_text,
        top_k=top_k
    )

    bm25_results = search_bm25(
        query_text=query_text,
        top_k=top_k
    )

    fused_results = (
        reciprocal_rank_fusion(
            faiss_results,
            bm25_results
        )
    )

    return fused_results[:top_k]