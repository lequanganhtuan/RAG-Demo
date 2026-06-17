from app.rag_service import RAGService
import json
import gradio as gr

service = RAGService()

# upload file
def process_file(file):
    if file is None:
        return "Please upload a file."

    result = service.process_document(file.name)

    if result["success"]:
        return (
            f"Ready: {result['file_name']}"
        )
    return result["message"]

def toggle_evaluation(enabled):
    service.enable_evaluation(enabled)
    return f"Evaluation: {enabled}"

def chat(message, history):
    result = service.ask(message)
    
    if not result["success"]:
        history.append(message, result["message"])
        return history, "", "", ""
    
    answer = result["answer"]
    
    history.append(
        (
            message,
            answer
        )
    )
    
    return (
        history,
        build_source_text(result),
        build_eval_text(result),
        build_debug_text(result)
    )
    
def build_source_text(result):
    sources = result["sources"]
    
    if not sources:
        return "No source"
    
    text = ""
    for source in sources:
        score = source.get(
            "rerank_score",
            source.get("score", 0)
        )
        
        text += (
            f"Chunk: {source['child_id']}\n" # retrival_result from search_faiss output 
            f"Score: {score:.3f}\n\n"
            f"{source['text'][:300]}\n"
            "------------------\n"
        )
        
    return text

def build_eval_text(result):
    evaluation = result["evaluation"]
    
    if evaluation is None:
        return "Evaluation disabled."

    return (
        f"Relevance: "
        f"{evaluation['relevance']}\n"

        f"Faithfulness: "
        f"{evaluation['faithfulness']}\n"

        f"Completeness: "
        f"{evaluation['completeness']}\n"

        f"Reason:\n"
        f"{evaluation['reason']}"
    )
    
def build_debug_text(result):
    debug = result["debug"]
    return json.dumps(
        debug,
        indent=2,
        ensure_ascii=False
    )
    

with gr.Blocks(
    title="RAG Assistant"
) as demo:
    gr.Markdown(
        "# RAG Assistant"
    )

    with gr.Row():
        # Left Panel
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload Document")
            process_btn = gr.Button("Process Document")
            status_box = gr.Textbox(label="Status")
            evaluation_checkbox = (
                gr.Checkbox(
                    label="Enable Evaluation"
                )
            )

        # Right Panel
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            user_input = gr.Textbox(label="Question")
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear History")

with gr.Accordion(
    "Sources",
    open=False
):
    source_box = gr.Textbox(lines=15)
    
with gr.Accordion(
    "Evaluation",
    open=False
):
    evaluation_box = gr.Textbox(lines=10)
    
with gr.Accordion(
    "Debug Information",
    open=False
):
    debug_box = gr.Textbox(lines=30)
    
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
        debug_box
    ]
)

def clear_chat():
    service.clear_history()
    return [], "", "", ""

clear_btn.click(
    clear_chat,
    outputs=[
        chatbot,
        source_box,
        evaluation_box,
        debug_box
    ]
)

if __name__ == "__main__":

    demo.launch()