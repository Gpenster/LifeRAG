import os
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-nano",
)

DB_NAME = os.getenv(
    "CHROMA_DB_DIR",
    "vector_db",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "4",
    )
)


# ============================================================
# Default personality
# ============================================================

DEFAULT_PERSONALITY = """
Respond in polished British English.

You are knowledgeable, confident and articulate.
You are helpful and concise while sounding natural rather than robotic.
"""


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """
You are George Penny's personal knowledge assistant.

Your job is to answer questions about George Penny using the
retrieved context supplied below.

IMPORTANT RULES:

1. Answer factual questions using only information supported by
   the supplied context.

2. If the information required to answer the question is not
   present in the context, say that you do not know.

3. Do not invent details about George.

4. You may combine information from multiple retrieved documents
   where appropriate.

5. You do not need to mention that you are using retrieved
   documents unless it is useful to the answer.

6. Answer the actual question directly rather than simply
   summarising the supplied context.

7. Follow the personality and response-style instructions below,
   but personality must NEVER override factual accuracy.


PERSONALITY AND RESPONSE STYLE:

{personality}


RETRIEVED CONTEXT:

{context}
"""


# ============================================================
# Vector store
# ============================================================

def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings,
    )


# ============================================================
# LLM
# ============================================================

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        temperature=0,
    )


# ============================================================
# Context formatting
# ============================================================

def format_context(
    docs: list[Any],
) -> str:
    """
    Format retrieved documents for the LLM.

    Note:
    This is the full context supplied to the model.

    The shorter context shown in the Streamlit UI is handled
    separately inside app.py.
    """

    parts = []

    for i, doc in enumerate(
        docs,
        start=1,
    ):
        source_file = doc.metadata.get(
            "source_file",
            "Unknown file",
        )

        page = doc.metadata.get(
            "page",
            "Unknown page",
        )

        if isinstance(page, int):
            display_page = page + 1
        else:
            display_page = page

        document_text = (
            doc.page_content.strip()
        )

        parts.append(
            f"[Document {i}]\n"
            f"Source file: {source_file}\n"
            f"Page: {display_page}\n"
            f"Content:\n"
            f"{document_text}"
        )

    return "\n\n".join(parts)


# ============================================================
# References
# ============================================================

def build_references(
    docs_with_scores: list[
        tuple[Any, float]
    ],
) -> list[dict]:

    references = []
    seen = set()

    for doc, score in docs_with_scores:

        source_file = doc.metadata.get(
            "source_file",
            "Unknown file",
        )

        page = doc.metadata.get(
            "page"
        )

        source = doc.metadata.get(
            "source",
            "",
        )

        snippet = (
            doc.page_content
            .strip()
            .replace("\n", " ")[:220]
        )

        key = (
            source_file,
            page,
            snippet,
        )

        if key in seen:
            continue

        seen.add(key)

        references.append(
            {
                "source_file": source_file,
                "page": (
                    page + 1
                    if isinstance(page, int)
                    else page
                ),
                "source": source,
                "score": round(
                    float(score),
                    4,
                ),
                "snippet": snippet,
            }
        )

    return references


# ============================================================
# Conversation history
# ============================================================

def build_history_messages(
    history: list[dict] | None,
) -> list[Any]:

    if not history:
        return []

    messages: list[Any] = []

    for item in history:

        if isinstance(item, dict):

            role = item.get("role")
            content = item.get(
                "content",
                "",
            )

            if not content:
                continue

            if role == "user":
                messages.append(
                    HumanMessage(
                        content=content
                    )
                )

            elif role == "assistant":
                messages.append(
                    AIMessage(
                        content=content
                    )
                )

            continue

        # Support older tuple/list history format
        if (
            isinstance(
                item,
                (list, tuple),
            )
            and item
        ):
            user_content = item[0]

            assistant_content = (
                item[1]
                if len(item) > 1
                else None
            )

            if user_content:
                messages.append(
                    HumanMessage(
                        content=str(
                            user_content
                        )
                    )
                )

            if assistant_content:
                messages.append(
                    AIMessage(
                        content=str(
                            assistant_content
                        )
                    )
                )

    return messages


# ============================================================
# Answer generation
# ============================================================

def answer_question(
    question: str,
    history=None,
    personality: str | None = None,
) -> dict:
    """
    Retrieve relevant context and generate an answer.

    `personality` affects only how the LLM responds.
    It does NOT affect the retrieval query.
    """

    if not isinstance(
        question,
        str,
    ):
        question = str(question)

    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    vectorstore = get_vectorstore()

    docs_with_scores = (
        vectorstore
        .similarity_search_with_relevance_scores(
            question,
            k=TOP_K,
        )
    )

    docs = [
        doc
        for doc, _ in docs_with_scores
    ]

    # --------------------------------------------------------
    # Prepare context
    # --------------------------------------------------------

    context = format_context(
        docs
    )

    # --------------------------------------------------------
    # Personality
    # --------------------------------------------------------

    effective_personality = (
        personality.strip()
        if personality
        else DEFAULT_PERSONALITY.strip()
    )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = (
        SYSTEM_PROMPT_TEMPLATE.format(
            context=context,
            personality=(
                effective_personality
            ),
        )
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    llm = get_llm()

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
        ]
        + build_history_messages(
            history
        )
        + [
            HumanMessage(
                content=question
            ),
        ]
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": str(
            response.content
        ),
        "context": docs,
        "references": (
            build_references(
                docs_with_scores
            )
        ),
    }