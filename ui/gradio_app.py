from app.rag_service import RAGService
import json
import gradio as gr

service = RAGService()


# =========================
# Document Processing
# =========================
def process_file(file):
    if file is None:
        return "Please upload a file."

    result = service.process_document(
        file.name
    )

    if result["success"]:
        return (
            f"Ready: {result['file_name']}"
        )

    return f"{result['message']}"


# =========================
# Evaluation Toggle
# =========================

def toggle_evaluation(enabled):
    service.enable_evaluation(enabled)
    return (
        f"Evaluation Enabled: {enabled}"
    )


# =========================
# Source Viewer
# =========================

def build_source_text(result):
    sources = result["sources"]
    if not sources:
        return "No sources found."
    text = ""
    
    for source in sources:
        score = source.get(
            "rerank_score",
            source.get("score", 0)
        )

        text += (
            f"Chunk ID: {source['child_id']}\n"
            f"Score: {score:.4f}\n\n"
            f"{source['text'][:500]}\n"
            f"\n{'='*50}\n\n"
        )

    return text


# =========================
# Evaluation Viewer
# =========================

def build_eval_text(result):
    evaluation = result["evaluation"]
    if evaluation is None:
        return "Evaluation disabled."

    return (
        f"Relevance: "
        f"{evaluation['relevance']}\n\n"

        f"Faithfulness: "
        f"{evaluation['faithfulness']}\n\n"

        f"Completeness: "
        f"{evaluation['completeness']}\n\n"

        f"Reason:\n"
        f"{evaluation['reason']}"
    )


# =========================
# Debug Viewer
# =========================

def build_debug_text(result):
    return json.dumps(
        result["debug"],
        indent=2,
        ensure_ascii=False
    )


# =========================
# Chat Handler
# =========================

def chat(message, history):
    result = service.ask(message)

    if not result["success"]:
        history.append(
            {
                "role": "user",
                "content": message
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": result["message"]
            }
        )
        return (
            history,
            "",
            "",
            "",
            ""
        )

    answer = result["answer"]
    history.append(
        {
            "role": "user",
            "content": message
        }
    )
    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return (
        history,
        build_source_text(result),
        build_eval_text(result),
        build_debug_text(result),
        ""
    )


# =========================
# Clear Chat
# =========================

def clear_chat():
    service.clear_history()
    return (
        [],
        "",
        "",
        "",
        ""
    )


# =========================
# UI
# =========================

with gr.Blocks(
    title="RAG Assistant"
) as demo:

    gr.Markdown(
        """
# RAG Assistant

Upload a document and ask questions about it.
"""
    )

    with gr.Row():
        # =====================
        # LEFT PANEL
        # =====================
        with gr.Column(scale=1):

            file_upload = gr.File(
                label="Upload Document"
            )

            process_btn = gr.Button(
                "Process Document"
            )

            status_box = gr.Textbox(
                label="Status",
                interactive=False
            )

            evaluation_checkbox = (
                gr.Checkbox(
                    label="Enable Evaluation"
                )
            )

        # =====================
        # RIGHT PANEL
        # =====================

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500,
            )
            user_input = gr.Textbox(
                label="Question",
                placeholder=(
                    "Ask a question..."
                )
            )

            with gr.Row():
                send_btn = gr.Button(
                    "Send"
                )
                clear_btn = gr.Button(
                    "Clear History"
                )

    # =====================
    # SOURCES PANEL
    # =====================

    with gr.Accordion(
        "Sources",
        open=False
    ):

        source_box = gr.Textbox(
            lines=15,
            interactive=False
        )

    # =====================
    # EVALUATION PANEL
    # =====================

    with gr.Accordion(
        "Evaluation",
        open=False
    ):
        evaluation_box = gr.Textbox(
            lines=10,
            interactive=False
        )

    # =====================
    # DEBUG PANEL
    # =====================

    with gr.Accordion(
        "Debug Information",
        open=False
    ):
        debug_box = gr.Textbox(
            lines=30,
            interactive=False
        )

    # =====================
    # EVENTS
    # =====================

    process_btn.click(
        process_file,
        inputs=file_upload,
        outputs=status_box
    )

    evaluation_checkbox.change(
        toggle_evaluation,
        inputs=evaluation_checkbox,
        outputs=status_box
    )

    send_btn.click(
        chat,
        inputs=[
            user_input,
            chatbot
        ],
        outputs=[
            chatbot,
            source_box,
            evaluation_box,
            debug_box,
            user_input
        ]
    )

    user_input.submit(
        chat,
        inputs=[
            user_input,
            chatbot
        ],
        outputs=[
            chatbot,
            source_box,
            evaluation_box,
            debug_box,
            user_input
        ]
    )

    clear_btn.click(
        clear_chat,
        outputs=[
            chatbot,
            source_box,
            evaluation_box,
            debug_box,
            user_input
        ]
    )


# =========================
# Launch
# =========================

if __name__ == "__main__":
    demo.queue()
    demo.launch()