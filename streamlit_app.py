import logging
import random
import uuid

import streamlit as st
from dotenv import load_dotenv

from implementation import db
from implementation.answer import answer_question
from implementation.sources import build_sources_summary

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# Label recorded alongside each logged interaction. The personality itself
# lives in implementation/answer.py (DEFAULT_PERSONALITY) — this is just
# the tag used for the DB record.
PERSONALITY = "posh_butler"

# A light, professional pirate theme — kept mostly to wording and emoji so
# the app still reads cleanly as a CV portfolio piece.
APP_TITLE = "🏴‍☠️ George Penny — Credit Risk, Analytics & Applied AI"
APP_TAGLINE = (
    "*Ask the crew about my career, projects, technical work or what "
    "I've been building.*"
)

LOADING_MESSAGES = [
    "🗺️ Checking the charts...",
    "⛵ Sailing through the archives...",
    "🧭 Searching the records...",
    "💰 Looking for useful treasure...",
]

SUGGESTED_QUESTIONS = [
    ("💼 Biggest career achievements", "What are George's biggest career achievements?"),
    ("🤖 AI projects", "What AI projects has George worked on?"),
    ("📊 Credit risk experience", "Tell me about George's credit risk experience."),
    ("🚴 Outside work", "What does George do outside work?"),
]


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


def render_sources_panel(docs):
    """Concise 'Sources & Evidence' citation view, not a raw chunk dump.

    Shows one friendly "Document — Section" (or "Document — Page N" for
    PDFs) label per unique retrieved chunk. Similarity scores and raw
    filesystem paths are never surfaced here; a short excerpt per source
    is available in the "View supporting evidence" expander for anyone
    who wants to check the grounding.
    """
    if not docs:
        st.caption(
            "🗺️ No treasure found yet — ask a question to see which "
            "parts of George's knowledge base were used."
        )
        return

    labels, excerpts = build_sources_summary(docs)

    for label in labels:
        st.markdown(f"- {label}")

    with st.expander("🧭 View supporting evidence"):
        for excerpt in excerpts:
            st.markdown(f"**{excerpt['label']}**")
            st.caption(excerpt["snippet"])


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
        logger.warning(
            "DB connection unavailable: %s", st.session_state.db_error
        )
        with st.expander(
            "⚠️ Database connection issue — chat logging is disabled",
            expanded=False,
        ):
            st.error(
                "Could not connect to the Supabase/PostgreSQL database. "
                "Answers will still work, but conversations are not being "
                "saved."
            )


def render_admin_sidebar():
    """Password-gated view of past sessions, tucked away in the sidebar."""
    with st.sidebar:
        with st.expander("Admin", expanded=False):
            admin_password = st.secrets.get("ADMIN_PASSWORD")
            if not admin_password:
                st.caption(
                    "Set ADMIN_PASSWORD in .streamlit/secrets.toml to "
                    "enable the history view."
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

            render_admin_history()


def render_admin_history():
    """Session/turn history view. Only called once the admin password checks out."""
    if st.session_state.get("db_error") is not None:
        st.error("Database is unavailable, so history cannot be loaded.")
        return

    try:
        sessions = db.fetch_sessions()
    except Exception:
        logger.exception("Failed to query session history")
        st.error("Failed to query session history.")
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
    except Exception:
        logger.exception("Failed to query session rows for %s", selected_session)
        st.error("Failed to query that session's rows.")
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="George Penny — Portfolio",
        page_icon="🏴‍☠️",
        layout="wide",
    )

    st.title(APP_TITLE)
    st.markdown(
        "*This is an AI-powered version of my professional portfolio. "
        "Rather than squeezing more than ten years of experience into "
        "two pages, you can ask it about my career, projects, technical "
        "work and leadership experience.*"
    )
    st.markdown(APP_TAGLINE)
    st.divider()

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "context" not in st.session_state:
        st.session_state.context = (
            "## Relevant Context\n\n_No context retrieved._"
        )

    if "source_docs" not in st.session_state:
        st.session_state.source_docs = []

    ensure_db_ready()
    render_db_status()
    render_admin_sidebar()

    # Two-column layout roughly matching the Gradio app
    chat_column, context_column = st.columns([1, 1])

    with chat_column:
        st.subheader("⚓ Ask the Crew")

        # Render existing conversation
        for message in st.session_state.messages:
            role = message["role"]
            content = normalize_content(message["content"])

            with st.chat_message(role):
                st.markdown(content)

        if not st.session_state.messages:
            st.caption("Not sure where to start?")
            suggestion_columns = st.columns(len(SUGGESTED_QUESTIONS))
            for column, (label, question) in zip(
                suggestion_columns, SUGGESTED_QUESTIONS
            ):
                with column:
                    if st.button(label, use_container_width=True):
                        st.session_state.pending_prompt = question
                        st.rerun()

        prompt = st.chat_input(
            "Ask about George's career, projects or adventures..."
        )

        if not prompt and st.session_state.get("pending_prompt"):
            prompt = st.session_state.pop("pending_prompt")

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
                with st.spinner(random.choice(LOADING_MESSAGES)):
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
                        st.session_state.source_docs = context

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
                            except Exception:
                                logger.exception("Failed to log interaction")
                                st.warning(
                                    "Answer was generated, but saving it "
                                    "to the database failed."
                                )

                    except Exception:
                        logger.exception("Failed to answer question")
                        st.error(
                            "🌊 Rough waters — something went wrong while "
                            "searching the records. Give it another try."
                        )

            # Needed so the context column refreshes immediately
            st.rerun()

    with context_column:
        st.subheader("🗺️ Sources & Evidence")

        with st.container(height=650, border=True):
            render_sources_panel(st.session_state.source_docs)

    st.divider()
    st.caption("⚓ Built with Python, Streamlit and RAG")


if __name__ == "__main__":
    main()
