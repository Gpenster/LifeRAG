import streamlit as st
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)


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
        return "\n".join(
            str(item) for item in value if item is not None
        )

    return str(value)


def format_context(context):
    if not context:
        return "## Relevant Context\n\n_No context retrieved._"

    lines = ["## Relevant Context", ""]

    for doc in context:
        source = (
            doc.metadata.get("source")
            or doc.metadata.get("source_file")
            or "Unknown source"
        )

        page = doc.metadata.get("page")

        if isinstance(page, int):
            page += 1

        lines.append(f"**Source:** {source}")

        if page is not None:
            lines.append(f"**Page:** {page}")

        lines.append("")
        lines.append(doc.page_content.strip())
        lines.append("")

    return "\n".join(lines)


def main():
    st.set_page_config(
        page_title="George Penny Knowledge Assistant",
        page_icon="💬",
        layout="wide",
    )

    st.title("George Penny Knowledge Assistant")
    st.caption("Ask questions about George Penny.")

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "context" not in st.session_state:
        st.session_state.context = (
            "## Relevant Context\n\n_No context retrieved._"
        )

    # Two-column layout roughly matching the Gradio app
    chat_column, context_column = st.columns([1, 1])

    with chat_column:
        st.subheader("Conversation")

        # Render existing conversation
        for message in st.session_state.messages:
            role = message["role"]
            content = normalize_content(message["content"])

            with st.chat_message(role):
                st.markdown(content)

        prompt = st.chat_input(
            "Ask anything about George Penny..."
        )

        if prompt:
            prompt = normalize_content(prompt)

            # Store and immediately render user message
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            # Everything before the latest user message becomes history
            prior_history = st.session_state.messages[:-1]

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = answer_question(
                            prompt,
                            prior_history,
                        )

                        answer_text = normalize_content(
                            result.get("answer")
                        )

                        context = result.get("context", [])

                        st.markdown(answer_text)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer_text,
                            }
                        )

                        st.session_state.context = format_context(
                            context
                        )

                    except Exception as exc:
                        st.error(
                            "Something went wrong while answering "
                            "the question."
                        )

                        # Useful while developing. You may want to
                        # remove this before exposing the app.
                        st.exception(exc)

            # Needed so the context column refreshes immediately
            st.rerun()

    with context_column:
        st.subheader("Retrieved Context")

        with st.container(height=650, border=True):
            st.markdown(st.session_state.context)


if __name__ == "__main__":
    main()
