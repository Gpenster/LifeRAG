"""Persistence layer for chat interactions, backed by Supabase Postgres.

Uses Streamlit's built-in SQL connection (`st.connection("postgresql", type="sql")`),
configured via `.streamlit/secrets.toml` -> [connections.postgresql] url.
"""
from __future__ import annotations

import streamlit as st
from sqlalchemy import text

CONNECTION_NAME = "postgresql"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context TEXT,
    personality TEXT
);
"""

_INSERT_SQL = """
INSERT INTO interactions (session_id, question, answer, context, personality)
VALUES (:session_id, :question, :answer, :context, :personality)
"""

_SELECT_SESSIONS_SQL = """
SELECT session_id, MIN(created_at) AS started_at, COUNT(*) AS turns
FROM interactions
GROUP BY session_id
ORDER BY started_at DESC
"""

_SELECT_SESSION_ROWS_SQL = """
SELECT interaction_id, session_id, created_at, question, answer, context, personality
FROM interactions
WHERE session_id = :session_id
ORDER BY created_at ASC
"""

_SELECT_RECENT_SQL = """
SELECT interaction_id, session_id, created_at, question, answer, context, personality
FROM interactions
ORDER BY created_at DESC
LIMIT :row_limit
"""


def get_connection():
    """Return the cached Streamlit SQL connection for Postgres."""
    return st.connection(CONNECTION_NAME, type="sql")


def test_connection() -> None:
    """Run a trivial query to confirm the DB is reachable. Raises on failure."""
    conn = get_connection()
    with conn.session as session:
        session.execute(text("SELECT 1"))


def init_db() -> None:
    """Create the interactions table if it does not already exist."""
    conn = get_connection()
    with conn.session as session:
        session.execute(text(_CREATE_TABLE_SQL))
        session.commit()


def log_interaction(
    session_id: str,
    question: str,
    answer: str,
    context: str,
    personality: str,
) -> None:
    """Insert exactly one row recording a single question/answer turn."""
    conn = get_connection()
    with conn.session as session:
        session.execute(
            text(_INSERT_SQL),
            {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "context": context,
                "personality": personality,
            },
        )
        session.commit()


def fetch_sessions():
    """Return one row per session_id with its start time and turn count."""
    conn = get_connection()
    return conn.query(_SELECT_SESSIONS_SQL, ttl=0)


def fetch_session_rows(session_id: str):
    """Return every interaction row for a single session, oldest first."""
    conn = get_connection()
    return conn.query(
        _SELECT_SESSION_ROWS_SQL,
        params={"session_id": session_id},
        ttl=0,
    )


def fetch_recent_interactions(row_limit: int = 200):
    """Return the most recent interactions across all sessions."""
    conn = get_connection()
    return conn.query(
        _SELECT_RECENT_SQL,
        params={"row_limit": row_limit},
        ttl=0,
    )
