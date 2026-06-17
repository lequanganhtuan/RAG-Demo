from pathlib import Path
from .loader import load_document
from .cleaner import clean_data
from .tokenizer_utils import token_length
from .chunker import recursive_chunking, SEPARATORS
from .parent_child import parent_child_chunking

# Threshold: ~20 pages × 500 tokens/page
# Use recursive chunking below this, parent-child above
TOKEN_THRESHOLD = 10_000

# This module return [vector_database, parent_store]

def run_pipeline(document_path: Path) -> tuple[list[dict], dict]:
    raw_text = load_document(document_path)
    clean_text = clean_data(raw_text)
    num_tokens = token_length(clean_text)

    print(f"Document: {num_tokens} tokens")

    if num_tokens < TOKEN_THRESHOLD:
        # Small doc — flat chunks, no parent needed
        chunks = recursive_chunking(clean_text, SEPARATORS)
        # called chunk (dont have vector database to save)
        vector_database = [
            {"child_id": f"chunk_{i}",
            "text": t,
            "parent_id": None,
            "vector": None}
            for i, t in enumerate(chunks)
        ]
        parent_store = {}
    else:
        # Large doc — parent-child for better context retrieval
        vector_database, parent_store = parent_child_chunking(clean_text, SEPARATORS)


    return vector_database, parent_store


# if __name__ == "__main__":
#     BASE_DIR = Path(__file__).parent
#     db, store = run_pipeline(BASE_DIR / "RAG_test.pdf")
#     print(f"Total chunks: {len(db)}, Parents: {len(store)}")