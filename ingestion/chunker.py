from loader import load_document
from cleaner import clean_data
from pathlib import Path
from transformers import AutoTokenizer

from tokenizer_utils import tokenizer, token_length

SEPARATORS = ["\n\n", "\n", ".", " "]

def token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def recursive_chunking(
    text: str,
    separators: list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> list[str]:
    """
    Recursively split text into chunks using a priority list of separators.
    Chunk size and overlap are measured in tokens, not characters.

    Args:
        text:         Input text to split.
        separators:   Ordered list of separators to try (e.g. ["\n\n", "\n", " "]).
        chunk_size:   Maximum tokens per chunk.
        chunk_overlap: Number of tokens to carry over into the next chunk.

    Returns:
        List of text chunks.
    """

    # Base case: text already fits in one chunk
    if token_length(text) <= chunk_size:
        return [text]

    # --- Fallback: no separators left, split by word ---
    # Avoids cutting mid-word unlike a raw character slice
    if not separators:
        chunks = []
        current_words = []

        for word in text.split():
            candidate = " ".join(current_words + [word])

            if token_length(candidate) <= chunk_size:
                # Word still fits — keep accumulating
                current_words.append(word)
            else:
                # Chunk is full — save it
                chunks.append(" ".join(current_words))

                # Build overlap by walking backwards until we hit the overlap budget
                overlap_words = []
                for w in reversed(current_words):
                    overlap_words.insert(0, w)
                    if token_length(" ".join(overlap_words)) >= chunk_overlap:
                        break

                # Start next chunk from overlap + current word
                current_words = overlap_words + [word]

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    # --- Recursive case: try the highest-priority separator first ---
    current_sep = separators[0]
    remaining_seps = separators[1:]

    # Separator not present — fall through to the next one
    if current_sep not in text:
        return recursive_chunking(text, remaining_seps, chunk_size, chunk_overlap)

    splits = text.split(current_sep)
    chunks = []
    current_chunk = ""

    for i, part in enumerate(splits):
        # Re-attach separator to all parts except the last,
        # preserving original document structure
        part_with_sep = part + current_sep if i < len(splits) - 1 else part

        if token_length(part_with_sep) > chunk_size:
            # Part is too large on its own — flush current chunk and recurse
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(
                recursive_chunking(part_with_sep, remaining_seps, chunk_size, chunk_overlap)
            )

        elif token_length(current_chunk + part_with_sep) <= chunk_size:
            # Part fits alongside what we already have — keep accumulating
            current_chunk += part_with_sep

        else:
            # Adding this part would overflow — save the current chunk first
            if current_chunk:
                chunks.append(current_chunk)

            # Carry over the last `chunk_overlap` tokens as context for the next chunk
            current_tokens = tokenizer.encode(current_chunk, add_special_tokens=False)
            overlap_tokens = current_tokens[-chunk_overlap:]
            overlap_text = tokenizer.decode(overlap_tokens, skip_special_tokens=True)

            current_chunk = overlap_text + part_with_sep

    # Save whatever remains after the loop
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

