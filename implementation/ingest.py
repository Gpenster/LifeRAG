"""Builds the Chroma vector store from GP_Knowledge_Base.

Loads two kinds of source material:
- PDFs (CV/LinkedIn): one Document per page, chunked by size only. Page
  numbers are the only structure available, so they're kept in metadata
  for citation (see implementation/sources.py).
- Markdown knowledge-base files (pliant.md, leadership.md, etc.): split on
  "#"/"##" headings first, so each chunk keeps its document title and
  section heading in metadata, then chunked by size only if a section is
  unusually long. This keeps each project/theme as a single coherent
  chunk wherever possible, rather than fragmenting a story across many
  tiny "Problem" / "Approach" / "Impact" pieces.

Run this script whenever the knowledge-base content changes and the
vector DB needs rebuilding (see README for the exact command).
"""
from __future__ import annotations

import glob
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "GP_Knowledge_Base")

DB_NAME = os.getenv("CHROMA_DB_DIR", "vector_db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# "#" -> document title, "##" -> section. "###" subheadings (Problem /
# Approach / Business impact, etc.) are deliberately left as plain text
# inside the chunk rather than split out, so a whole project stays
# together as one retrievable, citable unit.
MD_HEADERS_TO_SPLIT_ON = [
    ("#", "doc_title"),
    ("##", "section"),
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdf_documents() -> list[Document]:
    """Load CV/LinkedIn-style PDFs, one Document per page."""
    documents = []
    pdf_paths = glob.glob(
        os.path.join(KNOWLEDGE_BASE_DIR, "**", "*.pdf"), recursive=True
    )

    for file_path in sorted(pdf_paths):
        loader = PyPDFLoader(file_path)
        for doc in loader.load():
            doc.metadata["source_file"] = os.path.basename(file_path)
            doc.metadata["doc_type"] = "pdf"
            documents.append(doc)

    return documents


def load_markdown_documents() -> list[Document]:
    """Load GP_Knowledge_Base/*.md, splitting on headings so each chunk
    keeps its document title and section heading in metadata."""
    documents: list[Document] = []
    md_paths = sorted(glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md")))
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MD_HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )

    for file_path in md_paths:
        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()

        if not text:
            # Placeholder files that haven't been written yet are skipped
            # rather than indexed as empty chunks.
            continue

        source_file = os.path.basename(file_path)
        for split in header_splitter.split_text(text):
            split.metadata["source_file"] = source_file
            split.metadata["doc_type"] = "knowledge_base"
            documents.append(split)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Size-based safety net on top of the page/heading split above. Most
    markdown sections are already well under CHUNK_SIZE, so this mainly
    affects PDF pages and any unusually long section."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store() -> Chroma:
    pdf_docs = load_pdf_documents()
    md_docs = load_markdown_documents()

    print(
        f"Loaded {len(pdf_docs)} PDF page(s) and "
        f"{len(md_docs)} markdown section(s)"
    )

    chunks = chunk_documents(pdf_docs + md_docs)
    print(f"Split into {len(chunks)} chunk(s)")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME, embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )
    print(
        f"Vector store rebuilt with {vectorstore._collection.count()} "
        f"chunk(s) at '{DB_NAME}'"
    )
    return vectorstore


if __name__ == "__main__":
    build_vector_store()
