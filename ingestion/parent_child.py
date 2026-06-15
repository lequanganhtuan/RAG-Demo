from chunker import recursive_chunking, SEPARATORS

PARENT_SIZE = 1500
PARENT_OVERLAP = 100

def parent_child_chunking(
    text: str,
    separators: list[str] = SEPARATORS,
    child_size: int = 512,
    child_overlap: int = 50,
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
) -> tuple[list[dict], dict]:
    
    # Chunk the parent first with large size
    parent_chunks = recursive_chunking(
        text, separators,
        chunk_size=parent_size,
        chunk_overlap=parent_overlap
    )

    parent_store = {}
    vector_database = []

    # Chunking from parent chunk to get child chunk, and correspond to parent
    for parent_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"parent_{parent_idx}"
        parent_store[parent_id] = parent_text

        child_chunks = recursive_chunking(
            parent_text, separators,  
            chunk_size=child_size,
            chunk_overlap=child_overlap
        )

        for child_idx, child_text in enumerate(child_chunks):
            vector_database.append({
                "child_id": f"{parent_id}_child_{child_idx}",
                "text": child_text,
                "parent_id": parent_id,
                "vector": None  # Filled by embedder.py in pipeline
            })

    return vector_database, parent_store