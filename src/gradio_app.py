import gradio as gr
from .rag_pipeline import RAGPipeline

pipe = RAGPipeline()

def chat_fn(message, history):
    # stream the answer progressively in the UI
    out = pipe.answer(message, top_k_retrieve=12, top_k_rerank=5, stream=True)
    stream = out["stream"]
    contexts = out["contexts"]

    cite_text = "\n".join([f"• {c['doc_id']}: {c['text'][:120]}..." for c in contexts])

    running = ""
    for partial in stream:
        running = partial
        yield "", history + [[message, running]]

    final = f"{running}\n\n**Sources**\n{cite_text}"
    yield "", history + [[message, final]]

def build_app():
    with gr.Blocks(title="RAG Chatbot") as demo:
        gr.Markdown("# 🔎 RAG Chatbot (MS MARCO)\nType a question; the answer streams live.")
        chat = gr.Chatbot(height=480, show_copy_button=True)
        msg = gr.Textbox(placeholder="Ask a question…")
        clear = gr.Button("Clear")

        msg.submit(chat_fn, [msg, chat], [msg, chat])
        clear.click(lambda: ("", []), None, [msg, chat])

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()
