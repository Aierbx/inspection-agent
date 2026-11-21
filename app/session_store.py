import sqlite3
import json
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from app.config import Settings

settings = Settings()

def init_db() -> None:
    db_path = Path(settings.SESSION_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    title TEXT NOT NULL DEFAULT ''
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id)
            """)

def create_session() -> str:
    session_id = str(uuid4())
    with closing(sqlite3.connect(settings.SESSION_DB_PATH)) as conn:
        with conn:
            conn.execute(
                "INSERT INTO sessions (session_id) VALUES (?)",
                (session_id,),
            )
    return session_id


def session_exists(session_id: str) -> bool:
    with closing(sqlite3.connect(settings.SESSION_DB_PATH)) as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row is not None


def load_history(session_id: str) -> list[dict]:
    with closing(sqlite3.connect(settings.SESSION_DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT content_json FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def save_messages(session_id: str, messages: list[dict]) -> None:
    rows = [
        (session_id, message["role"], json.dumps(message, ensure_ascii=False))
        for message in messages
    ]
    with closing(sqlite3.connect(settings.SESSION_DB_PATH)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            conn.executemany(
                "INSERT INTO messages (session_id, role, content_json) VALUES (?, ?, ?)",
                rows,
            )
