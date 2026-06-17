from pathlib import Path

from retrieval.pipeline import build_retrieval_index
from retrieval.embedder import load_parent_store

from generation.pipeline import query_pipeline

from evaluation.judge import evaluate_answer


class RAGService:
    def __init__(self):

        # Chat history
        self.history = []

        # Document state
        self.document_loaded = False
        self.current_document = None
        self.document_path = None

        # Parent context store
        self.parent_store = {}

        # Evaluation mode
        self.evaluation_enabled = False

    def process_document(
        self,
        document_path: str | Path
    ) -> dict:

        try:

            document_path = Path(document_path)
            # Build vector database
            build_retrieval_index(document_path)

            # Load saved parent store
            self.parent_store = load_parent_store()

            # Save document info
            self.current_document = document_path.name

            self.document_path = str(document_path)

            # Reset history when loading new document
            self.history = []

            self.document_loaded = True

            return {
                "success": True,
                "status": "ready",
                "file_name": self.current_document
            }

        except Exception as e:

            return {
                "success": False,
                "status": "error",
                "message": str(e)
            }

    def ask(
        self,
        user_query: str
    ) -> dict:

        # Empty query
        if not user_query.strip():
            return {
                "success": False,
                "message": "Empty query."
            }

        # No document uploaded
        if not self.document_loaded:
            return {
                "success": False,
                "message": (
                    "Please upload and process a document first."
                )
            }

        try:
            rag_result = query_pipeline(
                user_query=user_query,
                parent_store=self.parent_store,
                history=self.history
            )

            answer = rag_result["answer"]

            # Update conversation history
            self.history.append(
                {
                    "role": "user",
                    "content": user_query
                }
            )

            self.history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            # Optional evaluation
            evaluation_result = None

            if self.evaluation_enabled:

                evaluation_result = (
                    evaluate_answer(
                        question=user_query,
                        context="\n\n".join(
                            rag_result["contexts"]
                        ),
                        answer=answer
                    )
                )

            return {
                "success": True,
                "answer": answer,
                # For source citation panel
                "sources":
                    rag_result[
                        "reranked_results"
                    ],

                # For evaluation panel
                "evaluation":
                    evaluation_result,

                # For debug panel
                "debug": {
                    "retrieval_results":
                        rag_result[
                            "retrieval_results"
                        ],

                    "reranked_results":
                        rag_result[
                            "reranked_results"
                        ],

                    "contexts":
                        rag_result[
                            "contexts"
                        ],

                    "system_prompt":
                        rag_result[
                            "system_prompt"
                        ],

                    "user_prompt":
                        rag_result[
                            "user_prompt"
                        ],

                    "history":
                        self.history
                }
            }

        except Exception as e:

            return {
                "success": False,
                "message": str(e)
            }

    def clear_history(self):

        self.history = []

        return {
            "success": True,
            "message": "History cleared."
        }

    def enable_evaluation(
        self,
        enabled: bool
    ):

        self.evaluation_enabled = enabled

        return {
            "success": True,
            "evaluation_enabled": enabled
        }

    def get_status(self):

        return {

            "document_loaded":
                self.document_loaded,

            "current_document":
                self.current_document,

            "document_path":
                self.document_path,

            "history_turns":
                len(self.history),

            "evaluation_enabled":
                self.evaluation_enabled
        }