from pathlib import Path

from loader import load_document
from cleaner import clean_data
from tokenizer_utils import token_length
from chunker import recursive_chunking, SEPARATORS
from parent_child import parent_child_chunking
from pipeline import run_pipeline

FILE_PATH = Path("ingestion/RAG_test.pdf")

print("\n" + "=" * 80)
print("STEP 1 - LOAD DOCUMENT")
print("=" * 80)

raw_text = load_document(FILE_PATH)

print("Load success")
print(f"Characters: {len(raw_text)}")
print(raw_text[:300])


print("\n" + "=" * 80)
print("STEP 2 - CLEAN DATA")
print("=" * 80)

clean_text = clean_data(raw_text)

print("Clean success")
print(f"Characters: {len(clean_text)}")
print(clean_text[:300])


print("\n" + "=" * 80)
print("STEP 3 - TOKEN COUNT")
print("=" * 80)

total_tokens = token_length(clean_text)

print(f"Total tokens: {total_tokens}")


print("\n" + "=" * 80)
print("STEP 4 - RECURSIVE CHUNKING")
print("=" * 80)

chunks = recursive_chunking(clean_text, SEPARATORS)

print(f"Total chunks: {len(chunks)}")

if chunks:
    print("\nFirst chunk:")
    print(chunks[0][:500])

    print("\nLast chunk:")
    print(chunks[-1][:500])


print("\n" + "=" * 80)
print("STEP 5 - PARENT CHILD")
print("=" * 80)

vector_db, parent_store = parent_child_chunking(clean_text)

print(f"Parents: {len(parent_store)}")
print(f"Children: {len(vector_db)}")

if vector_db:
    print("\nFirst child:")
    print(vector_db[0])


print("\n" + "=" * 80)
print("STEP 6 - FULL PIPELINE")
print("=" * 80)

final_db, final_store = run_pipeline(FILE_PATH)

print(f"Final chunks: {len(final_db)}")
print(f"Final parents: {len(final_store)}")

if final_db:
    print("\nFirst result:")
    print(final_db[0])

print("\nPipeline completed successfully.")