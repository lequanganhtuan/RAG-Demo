from sentence_transformers import CrossEncoder
from ingestion.tokenizer_utils import token_length

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_results(user_query: str, retrieval_results: list[dict], top_k: int = 3) -> list[dict]:
    if not retrieval_results:
        return []
    
    # create pairs (query, chunk)
    pairs = [
        (user_query, result["text"])
        for result in retrieval_results
    ]
    
    # Calculate score
    scores = reranker.predict(pairs)
    
    # Attach the score into result
    for result, score in zip(retrieval_results, scores):
        result["rerank_score"] = float(score)
    
    # sort score
    retrieval_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )
    
    return retrieval_results[:top_k]

def expand_parent_context(retrieval_results: list[dict],parent_store: dict) -> list[str]:
    contexts = []
    seen_parent_ids = set()

    for result in retrieval_results:
        parent_id = result.get("parent_id")

        # Recursive Chunking
        if parent_id is None:
            contexts.append(result["text"])

        # Parent-Child Chunking
        elif parent_id in parent_store:

            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                contexts.append(parent_store[parent_id])

    return contexts

def apply_token_budget(
    contexts: list[str],
    system_prompt: str,
    user_query: str,
    history: list[dict],
    max_model_tokens: int = 32000,
    reserved_output_tokens: int = 2000
) -> tuple[list[str], list[dict]]:
    
    # Input token
    available_input_tokens = (max_model_tokens - reserved_output_tokens)
    used_tokens = token_length(system_prompt) + token_length(user_query)
    
    # Context
    selected_contexts = []

    for context in contexts:
        context_tokens = token_length(context)
        if used_tokens + context_tokens > available_input_tokens:
            break
        selected_contexts.append(context)
        used_tokens += context_tokens
    
    # History
    selected_history = []

    for turn in reversed(history):
        turn_text = (
            f"{turn['role']}: {turn['content']}"
        )
        turn_tokens = token_length(turn_text)

        if used_tokens + turn_tokens > available_input_tokens:
            break
        # we scan by reverse the oldest scan first, when put into a list, the newest always get index = 0
        selected_history.insert(0, turn)
        used_tokens += turn_tokens

    return selected_contexts, selected_history