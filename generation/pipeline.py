from retrieval.retriever import search_faiss
from context_manager import (
    rerank_results,
    expand_parent_context,
    apply_token_budget
)
from prompt_builder import (
    build_prompt,
    SYSTEM_PROMPT
)
from generation.llm_router import call_api


def query_pipeline(
    user_query: str,
    parent_store: dict,
    history: list[dict]
):

    retrieval_results = search_faiss(
        query_text=user_query,
        top_k=20
    )

    reranked_results = rerank_results(
        user_query=user_query,
        retrieval_results=retrieval_results,
        top_k=5
    )

    contexts = expand_parent_context(
        reranked_results,
        parent_store
    )

    contexts, history = apply_token_budget(
        contexts=contexts,
        system_prompt=SYSTEM_PROMPT,
        user_query=user_query,
        history=history
    )

    system_prompt, user_prompt = build_prompt(
        contexts,
        history,
        user_query
    )

    answer = call_api(
        system_prompt,
        user_prompt,
        user_query=user_query
    )

    history.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return {
        "answer": answer,
        "history": history,
        "contexts": contexts,
        "retrieval_results": reranked_results
    }