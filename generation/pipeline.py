from retrieval.retriever import search_hybrid
from generation.context_manager import (
    rerank_results,
    expand_parent_context,
    apply_token_budget
)
from generation.prompt_builder import (
    build_prompt,
    SYSTEM_PROMPT
)
from generation.llm_router import call_api
from ingestion.tokenizer_utils import token_length


def query_pipeline(
    user_query: str,
    parent_store: dict,
    history: list[dict]
):
    print("1 SEARCH")
    retrieval_results = search_hybrid(
        query_text=user_query,
        top_k=20
    )   
    
    print(len(retrieval_results))
    
    print("2 RERANK")
    reranked_results = rerank_results(
        user_query=user_query,
        retrieval_results=retrieval_results,
        top_k=5
    )
    print("3 CONTEXT")

    contexts = expand_parent_context(
        reranked_results,
        parent_store
    )
    print("4 TOKEN BUDGET")
    contexts, selected_history = apply_token_budget(
        contexts=contexts,
        system_prompt=SYSTEM_PROMPT,
        user_query=user_query,
        history=history
    )
    print("CONTEXT COUNT:", len(contexts))

    for i, c in enumerate(contexts):
        print(
            f"Context {i}:",
            token_length(c)
        )
    
    
    print("5 PROMPT")
    system_prompt, user_prompt = build_prompt(
        contexts,
        selected_history,
        user_query
    )
    print("6 CALL API")
    answer = call_api(
        system_prompt,
        user_prompt,
        user_query=user_query
    )
    print("7 DONE")
    return {
        "answer": answer,
        "retrieval_results": retrieval_results,
        "reranked_results": reranked_results,
        "contexts": contexts,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "selected_history": selected_history
    }