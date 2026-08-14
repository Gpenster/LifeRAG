import uuid

import streamlit as st
from dotenv import load_dotenv

from implementation import db
from implementation.answer import answer_question

load_dotenv(override=True)

# There is currently only one system-prompt persona (see
# implementation/answer.py); this label is what gets recorded alongside
# each logged interaction so future personalities can be distinguished.
PERSONALITY = "talent_acquisition_manager"


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


def ensure_db_ready() -> bool:
    """Create the interactions table once per session and cache the result.

    This is intentionally cheap: it only hits the database once per
    browser session (not on every rerun), and it stays silent when the
    connection is healthy so it doesn't clutter the normal chat UI.
    """
    if "db_error" not in st.session_state:
        try:
            db.init_db()
            st.session_state.db_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            st.session_state.db_error = exc

    return st.session_state.db_error is None


def render_db_status():
    """Show a clear, collapsible error if the DB connection is broken.

    Renders nothing at all when the connection is healthy.
    """
    if st.session_state.get("db_error") is not None:
        with st.expander(
            "⚠️ Database connection issue — chat logging is disabled",
            expanded=False,
        ):
            st.error(
                "Could not connect to the Supabase/PostgreSQL database. "
                "Answers will still work, but conversations are not being "
                "saved. Check `.streamlit/secrets.toml` -> "
                "[connections.postgresql] url."
            )
            st.exception(st.session_state.db_error)


def render_admin_sidebar():
    """Password-gated view of past sessions, stored in the database."""
    with st.sidebar:
        st.subheader("Admin: Interaction History")

        admin_password = st.secrets.get("ADMIN_PASSWORD")
        if not admin_password:
            st.caption(
                "Set ADMIN_PASSWORD in .streamlit/secrets.toml to enable "
                "the history view."
            )
            return

        entered_password = st.text_input(
            "Admin password", type="password", key="admin_password_input"
        )

        if not entered_password:
            return

        if entered_password != admin_password:
            st.error("Incorrect password.")
            return

        if st.session_state.get("db_error") is not None:
            st.error("Database is unavailable, so history cannot be loaded.")
            st.exception(st.session_state.db_error)
            return

        try:
            sessions = db.fetch_sessions()
        except Exception as exc:
            st.error("Failed to query session history.")
            st.exception(exc)
            return

        if sessions.empty:
            st.caption("No interactions logged yet.")
            return

        st.caption(f"{len(sessions)} session(s) logged.")

        session_options = sessions["session_id"].tolist()
        selected_session = st.selectbox(
            "Session",
            session_options,
            format_func=lambda sid: (
                f"{sid[:8]}… — "
                f"{sessions.loc[sessions['session_id'] == sid, 'started_at'].iloc[0]} "
                f"({sessions.loc[sessions['session_id'] == sid, 'turns'].iloc[0]} turns)"
            ),
        )

        try:
            rows = db.fetch_session_rows(selected_session)
        except Exception as exc:
            st.error("Failed to query that session's rows.")
            st.exception(exc)
            return

        st.dataframe(rows, use_container_width=True, hide_index=True)


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

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "context" not in st.session_state:
        st.session_state.context = (
            "## Relevant Context\n\n_No context retrieved._"
        )

    ensure_db_ready()
    render_db_status()
    render_admin_sidebar()

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

                        # Persist exactly one row per successful answer.
                        # Logging failures must not be confused with RAG
                        # failures, and must not hide the answer that was
                        # already rendered above.
                        if st.session_state.get("db_error") is None:
                            try:
                                db.log_interaction(
                                    session_id=st.session_state.session_id,
                                    question=prompt,
                                    answer=answer_text,
                                    context=st.session_state.context,
                                    personality=PERSONALITY,
                                )
                            except Exception as db_exc:
                                st.warning(
                                    "Answer was generated, but saving it "
                                    "to the database failed."
                                )
                                st.exception(db_exc)

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
