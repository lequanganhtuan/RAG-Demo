import json

from generation.llm_router import call_api


JUDGE_SYSTEM_PROMPT = """
Bạn là chuyên gia đánh giá chất lượng hệ thống RAG.

Hãy chấm điểm theo 3 tiêu chí:

1. Relevance (0-5)
- 0: Không liên quan
- 3: Có liên quan nhưng chưa tập trung
- 5: Trả lời đúng trọng tâm câu hỏi

2. Faithfulness (0-5)
- 0: Bịa thông tin
- 3: Chủ yếu dựa vào context nhưng có suy diễn
- 5: Hoàn toàn dựa trên context

3. Completeness (0-5)
- 0: Thiếu hầu hết thông tin
- 3: Trả lời một phần
- 5: Bao phủ đầy đủ các ý quan trọng

Trả về JSON thuần.

Ví dụ:

{
    "relevance": 5,
    "faithfulness": 4,
    "completeness": 5,
    "reason": "..."
}

Không được trả markdown.
Không được thêm text ngoài JSON.
"""


def evaluate_answer(
    question: str,
    context: str,
    answer: str
) -> dict:

    judge_prompt = f"""
<Question>
{question}
</Question>

<Context>
{context}
</Context>

<Answer>
{answer}
</Answer>

Đánh giá theo tiêu chí đã mô tả.
Trả về JSON hợp lệ.
"""

    response = call_api(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=judge_prompt,
        user_query=question
    )

    try:
        return json.loads(response)

    except Exception:
        return {
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "reason": "Judge output parse failed"
        }