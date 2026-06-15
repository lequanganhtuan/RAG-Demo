from embedder import build_faiss_index, FAISS_INDEX_PATH, METADATA_PATH
import faiss
import pickle
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search_faiss(query_text: str, top_k: int = 20):
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Not find the data on the disk, run embedder.py first")
    
    # Load FAISS index and metadata
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata_store = pickle.load(f)
    
    
    # Encode and normalize query text
    query_vector = _model.encode(
        [query_text], normalize_embeddings=True
    ).astype("float32")
    
    scores, indices = index.search(query_vector, top_k)
    
    child_results = []
    # parent_contexts = []
    # seen_parent_ids = set() # Avoid duplicate
        
    # The results from child, using child to get the print parent
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:  
            metadata_info = metadata_store[idx]
            p_id = metadata_info["parent_id"]
            
            child_results.append({
                "score": float(score),
                "child_id": metadata_info["child_id"],
                "parent_id": p_id,
                "text": metadata_info["text"],
            })
            
            # if p_id is None: # Recursive
            #     parent_contexts.append(metadata_info["text"])
            # elif p_id in parent_store and p_id not in seen_parent_ids: # Parent-child 
            #     seen_parent_ids.add(p_id)
            #     parent_contexts.append(parent_store[p_id])
                
    # return {
    #     "child_details": child_results,
    #     "context_for_llm": "\n\n".join(parent_contexts) 
    # }
    
    return child_results