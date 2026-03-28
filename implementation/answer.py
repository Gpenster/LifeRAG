import os
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
DB_NAME = os.getenv("CHROMA_DB_DIR", "vector_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly talent acquisition manager representing George Penny.

Answer the user's question using only the context provided below.
If the answer is not in the context, say you do not know.
Be concise, helpful, and professional.

Context:
{context}
"""


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings,
    )


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        temperature=0,
    )


def format_context(docs: list[Any]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source_file = doc.metadata.get("source_file", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")
        text = doc.page_content.strip()

        parts.append(
            f"[Document {i}]\n"
            f"Source file: {source_file}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(parts)


def build_references(docs_with_scores: list[tuple[Any, float]]) -> list[dict]:
    references = []
    seen = set()

    for doc, score in docs_with_scores:
        source_file = doc.metadata.get("source_file", "Unknown file")
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "")
        snippet = doc.page_content.strip().replace("\n", " ")[:220]

        key = (source_file, page, snippet)
        if key in seen:
            continue
        seen.add(key)

        references.append(
            {
                "source_file": source_file,
                "page": page + 1 if isinstance(page, int) else page,
                "source": source,
                "score": round(float(score), 4),
                "snippet": snippet,
            }
        )

    return references


def build_history_messages(history: list[dict] | None) -> list[Any]:
    if not history:
        return []

    messages: list[Any] = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            continue

        if isinstance(item, (list, tuple)) and item:
            user_content = item[0]
            assistant_content = item[1] if len(item) > 1 else None
            if user_content:
                messages.append(HumanMessage(content=str(user_content)))
            if assistant_content:
                messages.append(AIMessage(content=str(assistant_content)))

    return messages


def answer_question(question: str, history=None) -> dict:
    if not isinstance(question, str):
        question = str(question)
    vectorstore = get_vectorstore()
    llm = get_llm()

    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
    docs = [doc for doc, _ in docs_with_scores]

    context = format_context(docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = llm.invoke(
        [SystemMessage(content=system_prompt)]
        + build_history_messages(history)
        + [HumanMessage(content=question)]
    )

    return {
        "answer": str(response.content),
        "context": docs,
        "references": build_references(docs_with_scores),
    }
