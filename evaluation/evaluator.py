import csv
from pathlib import Path
from statistics import mean

from generation.pipeline import query_pipeline
from evaluation.judge import evaluate_answer


EVAL_LOG_PATH = Path("evaluation_results.csv")


def run_batch_evaluation(
    questions: list[str],
    parent_store: dict
):
    history = []
    all_results = []

    for question in questions:
        rag_result = query_pipeline(
            user_query=question,
            parent_store=parent_store,
            history=history
        )

        
        answer = rag_result["answer"]

        context = "\n\n".join(
            rag_result["contexts"]
        )

        judge_result = evaluate_answer(
            question=question,
            context=context,
            answer=answer
        )

        row = {
            "question": question,
            "answer": answer,
            "relevance": judge_result["relevance"],
            "faithfulness": judge_result["faithfulness"],
            "completeness": judge_result["completeness"],
            "reason": judge_result["reason"]
        }

        all_results.append(row)

    save_results(all_results)

    return calculate_metrics(all_results)


def save_results(results):

    with open(
        EVAL_LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "answer",
                "relevance",
                "faithfulness",
                "completeness",
                "reason"
            ]
        )

        writer.writeheader()

        writer.writerows(results)
        
def calculate_metrics(results):
    return {
        "avg_relevance": round(
            mean(r["relevance"] for r in results),
            2
        ),

        "avg_faithfulness": round(
            mean(r["faithfulness"] for r in results),
            2
        ),

        "avg_completeness": round(
            mean(r["completeness"] for r in results),
            2
        ),

        "overall_score": round(
            mean(
                (
                    r["relevance"] * 0.2 +
                    r["faithfulness"] * 0.5 +
                    r["completeness"] * 0.3
                )
                for r in results
            ),
            2
        )
    }