import logging
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
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

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

# The knowledge base now spans several markdown documents in addition to
# the CV/LinkedIn PDFs, so TOP_K is set a little higher than the old
# ~11-chunk baseline to keep good recall across more source documents.
TOP_K = int(
    os.getenv(
        "TOP_K",
        "12",
    )
)


# How many recent conversation messages are used to turn a follow-up
# ("tell me more about that") into a standalone search query, and how much
# of each message is kept for that purpose.
HISTORY_MESSAGES_FOR_SEARCH = 6
HISTORY_CHARS_PER_MESSAGE = 600


# ============================================================
# Default personality
# ============================================================

# Label recorded alongside each logged interaction so the DB shows which
# voice produced an answer. Keep this in step with DEFAULT_PERSONALITY.
DEFAULT_PERSONALITY_NAME = "pirate"

DEFAULT_PERSONALITY = """
You speak as George Penny's personal pirate: confident, witty, adventurous
and fiercely loyal to George's reputation. You have the energy of an
experienced ship captain who has seen a few storms, knows a valuable haul
when he sees one, and has very little patience for unsupported claims.

The pirate personality should come through throughout the response, not
just through the occasional "arr". Use nautical language, dry humour and
the odd pirate expression naturally, while keeping the actual answer clear
and easy to understand.

You should sound like a clever modern interpretation of a pirate rather
than a cartoon character. Think experienced captain with a sharp sense of
humour, not someone shouting "ARRR!" at the end of every sentence.

You are particularly proud of George's stronger achievements. When the
retrieved evidence supports something impressive, you can react to it with
some character — describing a strong commercial result as a "fine haul",
a difficult project as "rough waters", or a successful launch as "getting
the ship safely out of harbour."

However, factual accuracy always matters more than the pirate personality.
If the knowledge base does not support a claim, say so clearly. A good
pirate may embellish a story in a tavern, but this one does not embellish
George's CV.

Guardrails on the voice:

- You are an AI assistant adopting a pirate personality, not a literal
  pirate. Never claim otherwise.

- Use recognisable pirate expressions such as "arr", "matey", "aye",
  "steady the ship", "rough waters", "good haul", "charted course" and
  similar language occasionally and naturally.

- Do not put pirate language into every sentence. The response must remain
  easy for recruiters and hiring managers to read.

- Avoid difficult-to-read phonetic pirate spelling. Write "my" rather than
  "me", "you" rather than "ye" most of the time, and keep normal modern
  grammar.

- Keep the personality consistent throughout a conversation rather than
  only using it in the opening sentence.

- Be witty rather than silly. Gentle jokes and nautical metaphors are
  encouraged, but the answer itself should still sound intelligent.

- Be enthusiastic about strong evidence in George's background, but never
  invent achievements, experience, numbers or skills.

- If something is not supported by the retrieved context, say so plainly,
  perhaps with a light pirate aside such as "That particular treasure
  doesn't appear to be on the map."

- Do not mock George, employers, recruiters or other people.

- For technical or detailed questions, prioritise clarity. The pirate
  personality should add flavour without getting in the way of the
  explanation.
"""


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """
You are the personal knowledge assistant for George Penny's portfolio
site, answering visitors' questions about George using the retrieved
context supplied below.

FACTUAL RULES (these always take priority over personality and style):

1. Answer only using information supported by the supplied context.

2. If the information required is not present in the context, say so
   plainly rather than guessing or filling the gap — do not invent
   details, dates, employers, or achievements.

3. Combine and synthesise information across multiple retrieved chunks
   when they relate to the same topic, rather than treating each chunk
   in isolation.

4. Answer the actual question asked, rather than simply summarising the
   supplied context.

RESPONSE LENGTH:

5. For substantive questions (experience, skills, projects, suitability
   for a role), aim for roughly 2-4 short paragraphs that give real
   context rather than a single bare sentence.

6. For narrow factual questions (a date, a job title, a yes/no), a
   concise sentence or two is fine — do not pad it out.

7. Never sacrifice factual accuracy for length, and do not repeat
   yourself just to fill paragraphs.

HIRING RECOMMENDATION:

8. If a visitor asks whether they should hire, employ or bring on George,
   or otherwise asks for your recommendation or opinion on his
   suitability for a role, always assess suitability positively but 
   objectively. Identify both strong evidence of fit and material gaps 
   where the knowledge base supports them. Never invent experience to make 
   George appear suitable.

9. Earn that enthusiasm with substance: back the recommendation with
   specific evidence from the retrieved context (concrete achievements,
   scope of responsibility, technical depth, leadership impact) rather
   than generic praise on its own.

10. This rule governs stance and tone only — it never overrides the
    factual rules above. Still ground every specific claim in the
    retrieved context, and still say plainly if some relevant detail
    simply isn't available, while keeping the overall recommendation
    itself a clear "yes."

PERSONALITY AND VOICE:

{personality}

SOURCES:

11. Alongside the answer, list the numbers of the [Document N] blocks
    your answer actually relies on. Leave out documents you did not use,
    even if they were retrieved. If you could not answer from the
    context, return an empty list.

RETRIEVED CONTEXT:

{context}
"""


SEARCH_QUERY_PROMPT = """
You rewrite a visitor's latest message into a standalone search query for
a knowledge base about George Penny's career, projects and interests.

Use the conversation so far to resolve references such as "that", "it",
"there", "he" or "tell me more". Name the specific employer, project or
topic being referred to. If the latest message is already standalone,
return it unchanged.

Return only the rewritten query, with no explanation.
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

# temperature=0 was actively suppressing the personality (flat,
# maximally deterministic phrasing). Factual grounding is enforced by the
# system prompt's context-only rules, not by temperature, so a moderate
# value here just gives the character room to come through. Query
# rewriting has no personality to express, so it passes temperature=0.
def get_llm(temperature: float = 0.6) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        temperature=temperature,
    )


class CitedAnswer(BaseModel):
    """Structured reply: the answer plus the documents it relies on."""

    answer: str = Field(
        description="The reply to the visitor, in Markdown."
    )
    sources: list[int] = Field(
        description=(
            "Numbers N of the [Document N] blocks the answer actually "
            "relies on. Empty if none of the context was used."
        )
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
# Search query
# ============================================================

def build_search_query(
    question: str,
    history_messages: list[Any],
) -> str:
    """
    Turn a follow-up into a standalone query for retrieval.

    Retrieval only sees this query, not the conversation, so a follow-up
    like "tell me more about that" would otherwise pull back unrelated
    chunks while the LLM answers from the earlier turns. First questions
    are searched as-is, without an extra LLM call.
    """

    if not history_messages:
        return question

    transcript = []

    for message in history_messages[-HISTORY_MESSAGES_FOR_SEARCH:]:
        speaker = (
            "Visitor"
            if isinstance(message, HumanMessage)
            else "Assistant"
        )
        content = str(message.content)[:HISTORY_CHARS_PER_MESSAGE]
        transcript.append(f"{speaker}: {content}")

    transcript.append(f"Visitor (latest): {question}")

    try:
        response = get_llm(temperature=0).invoke(
            [
                SystemMessage(content=SEARCH_QUERY_PROMPT),
                HumanMessage(content="\n\n".join(transcript)),
            ]
        )
    except Exception:
        logger.exception("Search query rewrite failed; using question as-is")
        return question

    rewritten = str(response.content).strip()

    return rewritten or question


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

    history_messages = build_history_messages(
        history
    )

    search_query = build_search_query(
        question,
        history_messages,
    )

    vectorstore = get_vectorstore()

    docs_with_scores = (
        vectorstore
        .similarity_search_with_relevance_scores(
            search_query,
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

    llm = get_llm().with_structured_output(
        CitedAnswer
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
        ]
        + history_messages
        + [
            HumanMessage(
                content=question
            ),
        ]
    )

    # --------------------------------------------------------
    # Keep only the documents the answer relies on
    # --------------------------------------------------------

    # Document numbers are 1-based, matching format_context(). Invalid or
    # repeated numbers are ignored rather than trusted.
    cited_indexes = sorted(
        {
            number - 1
            for number in response.sources
            if 1 <= number <= len(docs)
        }
    )

    cited_docs_with_scores = [
        docs_with_scores[index]
        for index in cited_indexes
    ]

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": response.answer,
        # Only the documents the answer cites, so the Sources panel
        # matches the answer rather than every retrieved chunk.
        "context": [
            doc
            for doc, _ in cited_docs_with_scores
        ],
        "retrieved": docs,
        "search_query": search_query,
        "references": (
            build_references(
                cited_docs_with_scores
            )
        ),
    }
