import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(data_dir) -> sqlite3.Connection:
    path = Path(data_dir) / "conversations.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def create_conversation(conn, slug: str) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO conversations (slug, title, created_at, updated_at) VALUES (?,?,?,?)",
        (slug, None, now, now),
    )
    conn.commit()
    return cur.lastrowid


def list_conversations(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, slug, title, updated_at FROM conversations ORDER BY updated_at DESC, id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conn, id: int) -> dict | None:
    row = conn.execute("SELECT id, slug, title FROM conversations WHERE id=?", (id,)).fetchone()
    if row is None:
        return None
    msgs = conn.execute(
        "SELECT role, content, citations_json FROM messages WHERE conversation_id=? ORDER BY id",
        (id,),
    ).fetchall()
    return {
        "id": row["id"], "slug": row["slug"], "title": row["title"],
        "messages": [
            {"role": m["role"], "content": m["content"],
             "citations": json.loads(m["citations_json"]) if m["citations_json"] else []}
            for m in msgs
        ],
    }


def add_message(conn, conversation_id: int, role: str, content: str, citations: list | None = None) -> None:
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, citations_json, created_at) VALUES (?,?,?,?,?)",
        (conversation_id, role, content, json.dumps(citations) if citations else None, _now()),
    )
    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conversation_id))
    conn.commit()


def set_title(conn, id: int, title: str) -> None:
    conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, id))
    conn.commit()


def get_history(conn, conversation_id: int, max_turns: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
        (conversation_id, max_turns * 2),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def delete_conversation(conn, id: int) -> None:
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (id,))
    conn.commit()
