from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(vector_database: list[dict]) -> list[dict]:
    """Fill the 'vector' field for each entry in vector_database."""
    # Get text and encode text
    texts = [entry["text"] for entry in vector_database]
    embeddings = _model.encode(texts, show_progress_bar=True)

    # get new key vector and get the embedddings list
    for entry, vector in zip(vector_database, embeddings):
        entry["vector"] = vector.tolist()

    return vector_database