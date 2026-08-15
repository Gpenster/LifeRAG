"""Helpers for turning raw retrieval metadata into recruiter-friendly source
labels for the "Sources & Evidence" panel.

Shared between the ingestion pipeline (implementation/ingest.py, which
writes source_file / doc_type / section metadata onto each chunk) and the
Streamlit UI (streamlit_app.py, which turns that metadata into citations).
"""
from __future__ import annotations

import os
import re

# Curated labels for known knowledge-base files, used verbatim when the
# filename matches. Anything not listed here falls back to a generic
# filename -> title conversion in friendly_source_name().
FRIENDLY_NAME_OVERRIDES = {
    "pliant.md": "Pliant",
    "klarna.md": "Klarna",
    "equifax.md": "Equifax",
    "stenn.md": "Stenn",
    "leadership.md": "Leadership",
    "technical_skills.md": "Technical Skills",
    "personal_projects_and_interests.md": "Personal Projects and Interests",
    "ai_projects.md": "AI Projects",
    "overview.md": "Professional Overview",
    "character.md": "About George",
    "credit_risk.md": "Credit Risk",
}


def friendly_source_name(source_file: str | None) -> str:
    """Map a raw filename to the recruiter-friendly label shown in the UI.

    Known knowledge-base files use a curated name from
    FRIENDLY_NAME_OVERRIDES. CV/LinkedIn PDFs keep their existing
    special-cased labels. Anything else falls back to a generic
    filename -> title conversion, e.g. "some_new_file.md" -> "Some New File",
    so newly added knowledge-base files still get a sensible label without
    code changes.
    """
    if not source_file:
        return "Unknown source"

    basename = os.path.basename(source_file)
    lowered = basename.lower()

    if lowered in FRIENDLY_NAME_OVERRIDES:
        return FRIENDLY_NAME_OVERRIDES[lowered]

    if "linkedin" in lowered:
        return "LinkedIn"

    if "cv" in lowered:
        return "CV"

    stem = os.path.splitext(basename)[0]
    words = re.split(r"[_\-\s]+", stem)
    title = " ".join(word.capitalize() for word in words if word)
    return title or basename


def get_section_label(metadata: dict) -> str | None:
    """Extract a human-readable section label from chunk metadata, if present.

    Prefers the section (H2) heading captured at ingestion time, falling
    back to the subsection (H3) heading. Returns None when neither is
    available (e.g. older PDF-derived chunks), so callers can fall back to
    the friendly document name instead.
    """
    section = metadata.get("section")
    if section:
        return str(section).strip()

    subsection = metadata.get("subsection")
    if subsection:
        return str(subsection).strip()

    return None


def build_source_label(metadata: dict) -> str:
    """Build the "Document — Section" citation label for a retrieved chunk.

    Markdown knowledge-base chunks prefer "Document — Section". PDF chunks
    (CV/LinkedIn) fall back to "Document — Page N", since page numbers are
    the only structure available there. If neither is available, just the
    friendly document name is returned rather than breaking the app.
    """
    source_file = metadata.get("source_file") or metadata.get("source")
    document = friendly_source_name(source_file)

    section = get_section_label(metadata)
    if section:
        return f"{document} — {section}"

    if metadata.get("doc_type") == "pdf":
        page = metadata.get("page_label")
        if page is None:
            raw_page = metadata.get("page")
            page = raw_page + 1 if isinstance(raw_page, int) else raw_page
        if page is not None:
            return f"{document} — Page {page}"

    return document


def build_sources_summary(docs):
    """Reduce retrieved chunks to a concise, deduplicated citation list.

    Returns (labels, excerpts):
    - labels: [str, ...] unique "Document — Section" style labels, in the
      order they were first seen (dedupes repeated chunks from the same
      section/page).
    - excerpts: [{"label": str, "snippet": str}, ...] one short excerpt per
      unique label, for the "View supporting evidence" expander.
    """
    if not docs:
        return [], []

    labels: list[str] = []
    excerpts = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata or {}
        label = build_source_label(metadata)

        if label in seen:
            continue
        seen.add(label)

        labels.append(label)

        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "…"
        excerpts.append({"label": label, "snippet": snippet})

    return labels, excerpts
