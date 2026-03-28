import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(context):
    if not context:
        return "## Relevant Context\n\n_No context retrieved._"

    lines = ["## Relevant Context", ""]
    for doc in context:
        source = doc.metadata.get("source") or doc.metadata.get("source_file") or "Unknown source"
        page = doc.metadata.get("page")
        if isinstance(page, int):
            page = page + 1

        lines.append(f"**Source:** {source}")
        if page is not None:
            lines.append(f"**Page:** {page}")
        lines.append("")
        lines.append(doc.page_content.strip())
        lines.append("")

    return "\n".join(lines)


def normalize_content(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return normalize_content(value.get("text"))
        return str(value)
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value if item is not None)
    return str(value)


def normalize_history(history):
    if not history:
        return []
    normalized = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = normalize_content(item.get("content"))
            if content:
                normalized.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and item:
            user_content = normalize_content(item[0])
            assistant_content = normalize_content(item[1] if len(item) > 1 else None)
            if user_content:
                normalized.append({"role": "user", "content": user_content})
            if assistant_content:
                normalized.append({"role": "assistant", "content": assistant_content})
    return normalized


def chat(history):
    if not history:
        return history, "## Relevant Context\n\n_No context retrieved._"
    normalized = normalize_history(history)
    if not normalized:
        return history, "## Relevant Context\n\n_No context retrieved._"
    last_message = normalized[-1]["content"]
    prior = normalized[:-1]
    result = answer_question(last_message, prior)
    answer_text = normalize_content(result.get("answer"))
    history.append({"role": "assistant", "content": answer_text})
    return history, format_context(result["context"])


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": normalize_content(message)}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="George Penny Knowledge Assistant") as ui:
        gr.Markdown("# George Penny Knowledge Assistant\nAsk questions about George Penny.")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about George Penny...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
