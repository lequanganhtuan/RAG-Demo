from retrieval.retriever import search_faiss

SYSTEM_PROMPT = """
Bạn là trợ lý hỏi đáp tài liệu.

Quy tắc:

1. Chỉ được sử dụng thông tin trong context.
2. Nếu không tìm thấy câu trả lời trong context, hãy trả lời:
    "Không tìm thấy thông tin trong tài liệu."
3. Không được suy diễn hoặc thêm kiến thức bên ngoài.
4. Trả lời bằng tiếng Việt.
5. Trích dẫn chunk liên quan nếu có.
"""

def build_prompt(contexts: list[str],history: list[dict],user_query: str) -> tuple[str, str]:
    # Build Context Section
    context_section = ""

    for idx, context in enumerate(contexts, start=1):
        context_section += f"""
<chunk id="{idx}">
{context}
</chunk>
"""

    # Build History Section
    history_section = ""

    for turn in history:
        history_section += (
            f"{turn['role']}: {turn['content']}\n"
        )

    user_prompt = f"""
<context>
{context_section}
</context>

<chat_history>
{history_section}
</chat_history>

<question>
{user_query}
</question>
"""

    return SYSTEM_PROMPT, user_prompt