import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .models import VideoMeta, Chunk, RunSummary


class VideoState(str, Enum):
    DISCOVERED = "discovered"
    TRANSCRIPT_FETCHED = "transcript_fetched"
    WHISPER_TRANSCRIBED = "whisper_transcribed"
    AD_STRIPPED = "ad_stripped"
    INDEXED = "indexed"
    FAILED_FETCH = "failed_fetch"
    NO_TRANSCRIPT = "no_transcript"
    FAILED_EMBED = "failed_embed"


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    duration INTEGER,
    upload_date TEXT,
    url TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'discovered',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_attempt_at TEXT,
    ad_method TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='rowid');
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    new INTEGER, done INTEGER, failed INTEGER, skipped INTEGER
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_video(conn: sqlite3.Connection, meta: VideoMeta) -> None:
    conn.execute(
        """INSERT INTO videos (video_id, title, duration, upload_date, url)
           VALUES (?,?,?,?,?)
           ON CONFLICT(video_id) DO UPDATE SET
             title=excluded.title, duration=excluded.duration,
             upload_date=excluded.upload_date, url=excluded.url""",
        (meta.video_id, meta.title, meta.duration, meta.upload_date, meta.url),
    )
    conn.commit()


def set_state(conn, video_id: str, state: VideoState, error: str | None = None) -> None:
    conn.execute(
        """UPDATE videos
           SET state=?, last_error=?, last_attempt_at=?, attempt_count=attempt_count+1
           WHERE video_id=?""",
        (state.value, error, _now(), video_id),
    )
    conn.commit()


def set_ad_method(conn, video_id: str, method: str) -> None:
    conn.execute("UPDATE videos SET ad_method=? WHERE video_id=?", (method, video_id))
    conn.commit()


def get_video(conn, video_id: str):
    return conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()


def videos_by_state(conn, states: list[VideoState]):
    placeholders = ",".join("?" for _ in states)
    return conn.execute(
        f"SELECT * FROM videos WHERE state IN ({placeholders}) ORDER BY upload_date DESC",
        [s.value for s in states],
    ).fetchall()


def count_by_state(conn) -> dict[str, int]:
    rows = conn.execute("SELECT state, COUNT(*) c FROM videos GROUP BY state").fetchall()
    return {r["state"]: r["c"] for r in rows}


def record_run(conn, summary: RunSummary, kind: str) -> None:
    conn.execute(
        "INSERT INTO runs (kind, started_at, new, done, failed, skipped) VALUES (?,?,?,?,?,?)",
        (kind, _now(), summary.new, summary.done, summary.failed, summary.skipped),
    )
    conn.commit()


def delete_video_chunks(conn, video_id: str) -> None:
    # FTS5 external-content table requires an explicit 'delete' per row before removing content rows.
    rows = conn.execute("SELECT rowid, text FROM chunks WHERE video_id=?", (video_id,)).fetchall()
    for r in rows:
        conn.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)", (r["rowid"], r["text"]))
    conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
    conn.commit()


def insert_chunks(conn, chunks: list[Chunk], title_of: dict[str, str]) -> None:
    for ch in chunks:
        cur = conn.execute(
            "INSERT INTO chunks (video_id, idx, start, title, text) VALUES (?,?,?,?,?)",
            (ch.video_id, ch.idx, ch.start, title_of[ch.video_id], ch.text),
        )
        conn.execute(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, ch.text)
        )
    conn.commit()
