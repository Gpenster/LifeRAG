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

# The knowledge base now spans several markdown documents in addition to
# the CV/LinkedIn PDFs, so TOP_K is set a little higher than the old
# ~11-chunk baseline to keep good recall across more source documents.
TOP_K = int(
    os.getenv(
        "TOP_K",
        "12",
    )
)


# ============================================================
# Default personality
# ============================================================

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

# temperature=0 was actively suppressing the butler personality (flat,
# maximally deterministic phrasing). Factual grounding is enforced by the
# system prompt's context-only rules, not by temperature, so a moderate
# value here just gives the character room to come through.
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        temperature=0.6,
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
