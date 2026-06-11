from transformers import AutoTokenizer

# Initialize once, import everywhere — avoid reloading the model multiple times
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))