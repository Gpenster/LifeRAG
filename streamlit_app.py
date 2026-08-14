import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

from implementation.answer import answer_question


load_dotenv()


# ============================================================
# Personality defaults
# ============================================================

DEFAULT_PERSONA = """
Respond as an exceptionally posh, upper-class British gentleman, slightly snobbish and dryly amused.

Tone:
- Polished, articulate and confident.
- Occasionally use understated British wit.
- Sound well educated and socially assured.
- You may gently imply that certain things are rather obvious,
  pedestrian or beneath serious consideration.
- Do not become rude, insulting or hostile.
- Do not overdo archaic language.
- Always prioritise answering the user's question accurately.

Keep responses relatively concise unless the question requires detail.
"""


# ============================================================
# Database
# ============================================================

def get_database_connection():
    return st.connection(
        "postgresql",
        type="sql",
    )


def initialise_database():
    """
    Creates the table automatically when the app starts.
    """
    conn = get_database_connection()

    with conn.session as session:
        session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id VARCHAR(36) PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context TEXT,
                    personality TEXT
                );
                """
            )
        )

        session.commit()


def save_interaction(
    session_id: str,
    question: str,
    answer: str,
    context: str,
    personality: str,
):
    """
    Persist a completed Q&A interaction.
    """
    conn = get_database_connection()

    with conn.session as session:
        session.execute(
            text(
                """
                INSERT INTO interactions (
                    interaction_id,
                    session_id,
                    question,
                    answer,
                    context,
                    personality
                )
                VALUES (
                    :interaction_id,
                    :session_id,
                    :question,
                    :answer,
                    :context,
                    :personality
                );
                """
            ),
            {
                "interaction_id": str(uuid.uuid4()),
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "context": context,
                "personality": personality,
            },
        )

        session.commit()


# ============================================================
# General helpers
# ============================================================

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
            str(item)
            for item in value
            if item is not None
        )

    return str(value)


def shorten_text(
    text_value: str,
    max_chars: int = 350,
) -> str:
    """
    Shortens retrieved context for display purposes only.

    The original retrieved context can still be used by the model.
    """
    text_value = " ".join(
        text_value.strip().split()
    )

    if len(text_value) <= max_chars:
        return text_value

    return text_value[:max_chars].rsplit(" ", 1)[0] + "..."


def format_context(
    context,
    max_documents: int = 3,
):
    """
    Creates a compact display of the evidence used.

    This does NOT alter the actual documents supplied to the RAG system.
    It only controls what is shown in the Streamlit UI.
    """
    if not context:
        return (
            "### Evidence used\n\n"
            "_No supporting context was retrieved._"
        )

    lines = [
        "### Evidence used",
        "",
        f"{len(context)} relevant source chunk(s) retrieved.",
        "",
    ]

    for index, doc in enumerate(
        context[:max_documents],
        start=1,
    ):
        source = (
            doc.metadata.get("source")
            or doc.metadata.get("source_file")
            or "Unknown source"
        )

        page = doc.metadata.get("page")

        if isinstance(page, int):
            page += 1

        lines.append(f"**{index}. {source}**")

        if page is not None:
            lines.append(f"Page {page}")

        lines.append("")
        lines.append(
            shorten_text(
                doc.page_content,
                max_chars=350,
            )
        )
        lines.append("")

    if len(context) > max_documents:
        hidden_count = (
            len(context) - max_documents
        )

        lines.append(
            f"_+ {hidden_count} additional "
            f"retrieved source chunk(s)._"
        )

    return "\n".join(lines)


# ============================================================
# Personality
# ============================================================

def build_personality_prompt(
    style: str,
    snobbery: int,
    response_length: str,
):
    """
    Build system-style instructions based on the UI settings.
    """

    if style == "Posh British":
        personality = DEFAULT_PERSONA

    elif style == "Professional":
        personality = """
Respond professionally, clearly and confidently.
Use natural British English.
Avoid unnecessary jargon.
"""

    elif style == "Friendly":
        personality = """
Respond warmly and conversationally.
Use natural British English and keep the tone approachable.
"""

    else:
        personality = ""

    if snobbery <= 1:
        personality += """
Keep any snobbishness extremely subtle.
"""

    elif snobbery == 2:
        personality += """
Use occasional understated upper-class superiority,
but remain pleasant.
"""

    elif snobbery == 3:
        personality += """
Be noticeably posh and slightly snobbish.
Use dry wit and occasional gentle condescension.
"""

    elif snobbery == 4:
        personality += """
Lean strongly into the upper-class British persona.
Be distinctly snobbish and dryly judgemental,
while remaining amusing rather than unpleasant.
"""

    else:
        personality += """
Adopt an exaggeratedly posh, socially superior British persona.
Treat mundane matters with faint bemusement and elegant disdain.
Remain helpful and never directly insult the user.
"""

    personality += f"""

Preferred response length: {response_length}.
"""

    return personality.strip()


# ============================================================
# Admin interface
# ============================================================

def render_admin():
    st.subheader("Conversation History")

    st.caption(
        "Review anonymous sessions and questions submitted "
        "through the app."
    )

    configured_password = st.secrets.get(
        "ADMIN_PASSWORD"
    )

    if not configured_password:
        st.warning(
            "ADMIN_PASSWORD has not been configured "
            "in Streamlit Secrets."
        )
        return

    password = st.text_input(
        "Admin password",
        type="password",
    )

    if not password:
        return

    if password != configured_password:
        st.error("Incorrect password.")
        return

    conn = get_database_connection()

    # ttl=0 prevents Streamlit from returning a stale cached query.
    interactions = conn.query(
        """
        SELECT
            interaction_id,
            session_id,
            created_at,
            question,
            answer,
            context,
            personality
        FROM interactions
        ORDER BY created_at DESC;
        """,
        ttl=0,
    )

    if interactions.empty:
        st.info(
            "No conversations have been recorded yet."
        )
        return

    total_questions = len(interactions)

    total_sessions = (
        interactions["session_id"]
        .nunique()
    )

    average_questions = (
        total_questions / total_sessions
        if total_sessions
        else 0
    )

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Sessions",
        total_sessions,
    )

    metric_2.metric(
        "Questions",
        total_questions,
    )

    metric_3.metric(
        "Questions / session",
        f"{average_questions:.1f}",
    )

    st.divider()

    st.subheader("Sessions")

    session_ids = (
        interactions["session_id"]
        .drop_duplicates()
        .tolist()
    )

    for session_number, session_id in enumerate(
        session_ids,
        start=1,
    ):
        session_data = interactions[
            interactions["session_id"]
            == session_id
        ].sort_values("created_at")

        first_time = session_data.iloc[0][
            "created_at"
        ]

        label = (
            f"Session {session_number} — "
            f"{first_time} — "
            f"{len(session_data)} question(s)"
        )

        with st.expander(label):

            st.caption(
                f"Session ID: {session_id}"
            )

            for _, row in session_data.iterrows():

                st.markdown(
                    f"**User:** {row['question']}"
                )

                st.markdown(
                    f"**Assistant:** {row['answer']}"
                )

                with st.expander(
                    "Retrieved evidence",
                ):
                    st.markdown(
                        row["context"]
                        or "_None stored._"
                    )

                st.divider()


# ============================================================
# Main application
# ============================================================

def main():
    st.set_page_config(
        page_title=(
            "George Penny Knowledge Assistant"
        ),
        page_icon="💬",
        layout="wide",
    )

    initialise_database()

    st.title(
        "George Penny Knowledge Assistant"
    )

    st.caption(
        "Ask questions about George Penny."
    )

    # --------------------------------------------------------
    # Personality settings
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Assistant settings",
        expanded=False,
    ):
        setting_1, setting_2, setting_3 = (
            st.columns(3)
        )

        with setting_1:
            personality_style = st.selectbox(
                "Personality",
                [
                    "Posh British",
                    "Professional",
                    "Friendly",
                ],
                index=0,
            )

        with setting_2:
            snobbery = st.slider(
                "Snobbery",
                min_value=0,
                max_value=5,
                value=3,
                help=(
                    "0 = perfectly polite. "
                    "5 = magnificently superior."
                ),
            )

        with setting_3:
            response_length = st.selectbox(
                "Response length",
                [
                    "Concise",
                    "Medium",
                    "Detailed",
                ],
                index=0,
            )

    personality_prompt = (
        build_personality_prompt(
            style=personality_style,
            snobbery=snobbery,
            response_length=response_length,
        )
    )

    st.caption(
        "Conversations are stored anonymously "
        "to help review and improve the assistant."
    )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "context" not in st.session_state:
        st.session_state.context = (
            "### Evidence used\n\n"
            "_No context retrieved yet._"
        )

    if "session_id" not in st.session_state:
        st.session_state.session_id = (
            str(uuid.uuid4())
        )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    chat_tab, admin_tab = st.tabs(
        [
            "💬 Chat",
            "🔒 Admin",
        ]
    )

    # --------------------------------------------------------
    # Chat
    # --------------------------------------------------------

    with chat_tab:

        chat_column, context_column = (
            st.columns(
                [1.4, 0.6]
            )
        )

        with chat_column:
            st.subheader("Conversation")

            for message in (
                st.session_state.messages
            ):
                role = message["role"]

                content = normalize_content(
                    message["content"]
                )

                with st.chat_message(role):
                    st.markdown(content)

            prompt = st.chat_input(
                "Ask anything about George Penny..."
            )

            if prompt:
                prompt = normalize_content(
                    prompt
                )

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt,
                    }
                )

                with st.chat_message("user"):
                    st.markdown(prompt)

                prior_history = (
                    st.session_state.messages[:-1]
                )

                with st.chat_message(
                    "assistant"
                ):
                    with st.spinner(
                        "Considering the matter..."
                    ):
                        try:

                            result = answer_question(
                                prompt,
                                prior_history,
                                personality=(
                                    personality_prompt
                                ),
                            )

                            answer_text = (
                                normalize_content(
                                    result.get(
                                        "answer"
                                    )
                                )
                            )

                            context = result.get(
                                "context",
                                [],
                            )

                            context_text = (
                                format_context(
                                    context
                                )
                            )

                            st.markdown(
                                answer_text
                            )

                            st.session_state.messages.append(
                                {
                                    "role": (
                                        "assistant"
                                    ),
                                    "content": (
                                        answer_text
                                    ),
                                }
                            )

                            st.session_state.context = (
                                context_text
                            )

                            save_interaction(
                                session_id=(
                                    st.session_state
                                    .session_id
                                ),
                                question=prompt,
                                answer=answer_text,
                                context=context_text,
                                personality=(
                                    personality_prompt
                                ),
                            )

                        except Exception as exc:
                            st.error(
                                "Something went wrong "
                                "while answering the "
                                "question."
                            )

                            st.exception(exc)

                st.rerun()

        with context_column:
            st.subheader(
                "Evidence"
            )

            with st.container(
                height=500,
                border=True,
            ):
                st.markdown(
                    st.session_state.context
                )

    # --------------------------------------------------------
    # Admin
    # --------------------------------------------------------

    with admin_tab:
        render_admin()


if __name__ == "__main__":
    main()