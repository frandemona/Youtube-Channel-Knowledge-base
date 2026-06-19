# YouTube Channel Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI + web tool that ingests a YouTube channel's transcripts, strips sponsor reads, indexes them, and answers questions per channel via an agentic RAG assistant with citations.

**Architecture:** A Python package `ytkb` holds all logic; a Typer CLI and a FastAPI web page are thin layers over it. Each channel is an isolated folder under `data/channels/<slug>/` with a SQLite state/log/FTS5 DB, raw + cleaned transcripts on disk, and a LanceDB vector table. Ingestion is an incremental, resumable per-video state machine; querying is a tool-using agent over local hybrid search (FTS5 + embeddings).

**Tech Stack:** Python 3.12, `uv`, yt-dlp, requests (SponsorBlock), faster-whisper, fastembed (`bge-small-en-v1.5`), LanceDB, SQLite+FTS5 (stdlib), OpenRouter via the `openai` SDK, Typer, FastAPI + uvicorn, pytest.

## Global Constraints

- Python **3.12**; manage the project with **uv** (`uv init`, `uv add`, `uv run`).
- Package name is **`ytkb`**; CLI entry point is **`kb`**.
- LLM calls go through **OpenRouter** (`base_url=https://openrouter.ai/api/v1`) using the **`openai`** SDK; the model id is read from config and is swappable. Default chat/ad-strip model: **`anthropic/claude-haiku-4.5`** (tool-calling capable).
- Embeddings are **local** via fastembed model **`BAAI/bge-small-en-v1.5`** (384 dims). No embedding API calls.
- Config from `data/config.toml` (read with stdlib **`tomllib`**) + secrets from `data/.env` (`OPENROUTER_API_KEY`).
- Read config from `tomllib` only; never add a TOML-writing dependency. Channel metadata is JSON.
- All data is local under a single `data/` root (default `./data`, overridable via `YTKB_DATA_DIR`). Never commit `data/` or `.env` (already in `.gitignore`).
- Every task is TDD: write the failing test, see it fail, implement minimally, see it pass, commit. Network/LLM access is mocked in tests.

---

## File Structure

```
pyproject.toml                 # uv project + deps + [project.scripts] kb = "ytkb.cli:app"
ytkb/
  __init__.py
  config.py                    # Config dataclass + load_config()
  paths.py                     # data-dir layout, ChannelPaths, slugify, list_channel_slugs
  models.py                    # Segment, VideoMeta, Chunk, ChunkHit, Citation, Answer, RunSummary
  db.py                        # SQLite schema, connect/init, VideoState, state + log helpers
  channel.py                   # resolve_channel(), ChannelFilters, list_videos()  (yt-dlp)
  transcripts.py               # fetch_captions(), save_raw/load_raw  (yt-dlp json3)
  whisper_fallback.py          # transcribe()  (faster-whisper)
  sponsorblock.py              # get_segments(), strip_segments()
  llm.py                       # LLMClient (OpenRouter): complete(), chat_with_tools()
  adstrip.py                   # strip_ads()  (SponsorBlock -> LLM fallback)
  chunking.py                  # chunk_segments()
  embeddings.py                # Embedder (fastembed)
  store.py                     # ChannelStore: add(), keyword_search, semantic_search, read_around, list_videos
  retrieval.py                 # build_tools(store) -> (tool_specs, dispatch)
  agent.py                     # answer()  (agentic tool loop -> Answer)
  pipeline.py                  # ChannelContext, process_video()
  sync.py                      # add_channel, sync_channel, retry_channel, channel_status
  cli.py                       # Typer app: add/sync/ask/status/retry/list/web/remove
  web/
    app.py                     # FastAPI app + SSE streaming
    static/index.html          # single-page chat UI
tests/
  conftest.py                  # fixtures: tmp data dir, sample segments, fake LLM
  test_*.py                    # one per module
```

---

### Task 1: Project scaffold, config, and paths

**Files:**
- Create: `pyproject.toml`, `ytkb/__init__.py`, `ytkb/config.py`, `ytkb/paths.py`
- Test: `tests/test_config.py`, `tests/test_paths.py`, `tests/conftest.py`

**Interfaces:**
- Produces:
  - `ytkb.config.Config` dataclass with fields: `data_dir: Path`, `openrouter_api_key: str | None`, `chat_model: str`, `adstrip_model: str`, `embedding_model: str`, `chunk_tokens: int`, `chunk_overlap: int`, `languages: list[str]`, `whisper_enabled_default: bool`, `whisper_model: str`, `request_delay: float`, `top_k: int`.
  - `ytkb.config.load_config(data_dir: Path | None = None) -> Config`
  - `ytkb.paths.slugify(name: str) -> str`
  - `ytkb.paths.channels_root(data_dir: Path) -> Path`
  - `ytkb.paths.list_channel_slugs(data_dir: Path) -> list[str]`
  - `ytkb.paths.ChannelPaths` with attrs `dir, channel_json, db, transcripts_dir, vectors_dir` and methods `raw_path(video_id) -> Path`, `clean_path(video_id) -> Path`; constructed via `ChannelPaths.for_slug(data_dir, slug)`.

- [ ] **Step 1: Initialize the uv project and add dependencies**

Run:
```bash
uv init --package --name ytkb --python 3.12 .
uv add yt-dlp requests fastembed lancedb typer fastapi "uvicorn[standard]" openai faster-whisper python-dotenv
uv add --dev pytest
```
Then edit `pyproject.toml` to add the CLI entry point under `[project.scripts]`:
```toml
[project.scripts]
kb = "ytkb.cli:app"
```
Expected: `uv run python -c "import ytkb"` succeeds.

- [ ] **Step 2: Write the failing test for paths**

`tests/test_paths.py`:
```python
from pathlib import Path
from ytkb.paths import slugify, ChannelPaths, channels_root, list_channel_slugs


def test_slugify_normalizes():
    assert slugify("Y Combinator!") == "y-combinator"
    assert slugify("@MyChannel") == "mychannel"


def test_channel_paths_layout(tmp_path: Path):
    cp = ChannelPaths.for_slug(tmp_path, "ycombinator")
    assert cp.dir == tmp_path / "channels" / "ycombinator"
    assert cp.channel_json == cp.dir / "channel.json"
    assert cp.db == cp.dir / "videos.db"
    assert cp.raw_path("abc123") == cp.transcripts_dir / "abc123.raw.json"
    assert cp.clean_path("abc123") == cp.transcripts_dir / "abc123.clean.txt"


def test_list_channel_slugs(tmp_path: Path):
    (channels_root(tmp_path) / "a").mkdir(parents=True)
    (channels_root(tmp_path) / "b").mkdir(parents=True)
    assert list_channel_slugs(tmp_path) == ["a", "b"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ytkb.paths'`

- [ ] **Step 4: Implement `ytkb/paths.py`**

```python
import re
from dataclasses import dataclass
from pathlib import Path


def slugify(name: str) -> str:
    name = name.strip().lower().lstrip("@")
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def channels_root(data_dir: Path) -> Path:
    return Path(data_dir) / "channels"


def list_channel_slugs(data_dir: Path) -> list[str]:
    root = channels_root(data_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@dataclass(frozen=True)
class ChannelPaths:
    dir: Path

    @classmethod
    def for_slug(cls, data_dir: Path, slug: str) -> "ChannelPaths":
        return cls(channels_root(data_dir) / slug)

    @property
    def channel_json(self) -> Path:
        return self.dir / "channel.json"

    @property
    def db(self) -> Path:
        return self.dir / "videos.db"

    @property
    def transcripts_dir(self) -> Path:
        return self.dir / "transcripts"

    @property
    def vectors_dir(self) -> Path:
        return self.dir / "vectors"

    def raw_path(self, video_id: str) -> Path:
        return self.transcripts_dir / f"{video_id}.raw.json"

    def clean_path(self, video_id: str) -> Path:
        return self.transcripts_dir / f"{video_id}.clean.txt"

    def ensure(self) -> None:
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Write the failing test for config**

`tests/test_config.py`:
```python
from pathlib import Path
from ytkb.config import load_config, Config


def test_load_config_defaults(tmp_path: Path):
    cfg = load_config(tmp_path)
    assert isinstance(cfg, Config)
    assert cfg.data_dir == tmp_path
    assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.chat_model == "anthropic/claude-haiku-4.5"
    assert cfg.chunk_tokens == 500
    assert cfg.top_k == 6


def test_load_config_reads_toml_and_env(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        'chat_model = "openai/gpt-5-mini"\nchunk_tokens = 300\n'
    )
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-test-123\n")
    cfg = load_config(tmp_path)
    assert cfg.chat_model == "openai/gpt-5-mini"
    assert cfg.chunk_tokens == 300
    assert cfg.openrouter_api_key == "sk-test-123"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ytkb.config'`

- [ ] **Step 7: Implement `ytkb/config.py`**

```python
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

DEFAULTS = {
    "chat_model": "anthropic/claude-haiku-4.5",
    "adstrip_model": "anthropic/claude-haiku-4.5",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "chunk_tokens": 500,
    "chunk_overlap": 80,
    "languages": ["en"],
    "whisper_enabled_default": True,
    "whisper_model": "base",
    "request_delay": 1.0,
    "top_k": 6,
}


@dataclass
class Config:
    data_dir: Path
    openrouter_api_key: str | None
    chat_model: str
    adstrip_model: str
    embedding_model: str
    chunk_tokens: int
    chunk_overlap: int
    languages: list[str]
    whisper_enabled_default: bool
    whisper_model: str
    request_delay: float
    top_k: int


def default_data_dir() -> Path:
    return Path(os.environ.get("YTKB_DATA_DIR", "data")).resolve()


def load_config(data_dir: Path | None = None) -> Config:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    values = dict(DEFAULTS)
    toml_path = data_dir / "config.toml"
    if toml_path.exists():
        with toml_path.open("rb") as f:
            values.update(tomllib.load(f))

    env = dotenv_values(data_dir / ".env")
    api_key = env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

    return Config(data_dir=data_dir, openrouter_api_key=api_key, **values)
```

- [ ] **Step 8: Add shared fixtures in `tests/conftest.py`**

```python
import pytest
from pathlib import Path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: PASS (4 tests)

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock ytkb/ tests/
git commit -m "feat: project scaffold, config loading, and path layout"
```

---

### Task 2: SQLite schema, state machine, and log helpers

**Files:**
- Create: `ytkb/models.py`, `ytkb/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `ytkb.paths.ChannelPaths`.
- Produces:
  - `ytkb.models`: `@dataclass Segment(start: float, end: float, text: str)`; `@dataclass VideoMeta(video_id: str, title: str, duration: int | None, upload_date: str | None, url: str)`; `@dataclass Chunk(video_id: str, idx: int, start: float, text: str)`; `@dataclass ChunkHit(video_id: str, title: str, start: float, text: str, score: float)`; `@dataclass Citation(video_id: str, title: str, start: float, url: str)`; `@dataclass Answer(text: str, citations: list[Citation])`; `@dataclass RunSummary(new: int, done: int, failed: int, skipped: int, errors: list[str])`.
  - `ytkb.db.VideoState` (str Enum): `DISCOVERED, TRANSCRIPT_FETCHED, WHISPER_TRANSCRIBED, AD_STRIPPED, INDEXED, FAILED_FETCH, NO_TRANSCRIPT, FAILED_EMBED`.
  - `ytkb.db.connect(db_path: Path) -> sqlite3.Connection` (creates schema if missing).
  - `upsert_video(conn, meta: VideoMeta) -> None` (inserts as DISCOVERED if new, never downgrades state).
  - `set_state(conn, video_id: str, state: VideoState, error: str | None = None) -> None` (increments `attempt_count`, sets `last_attempt_at`).
  - `get_video(conn, video_id) -> sqlite3.Row | None`
  - `videos_by_state(conn, states: list[VideoState]) -> list[sqlite3.Row]`
  - `count_by_state(conn) -> dict[str, int]`
  - `record_run(conn, summary: RunSummary, kind: str) -> None`
  - `insert_chunks(conn, chunks: list[Chunk], title_of: dict[str, str]) -> None` (writes `chunks` + FTS5).

- [ ] **Step 1: Write `ytkb/models.py`** (no test needed — pure dataclasses, exercised by later tests)

```python
from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class VideoMeta:
    video_id: str
    title: str
    duration: int | None
    upload_date: str | None
    url: str


@dataclass
class Chunk:
    video_id: str
    idx: int
    start: float
    text: str


@dataclass
class ChunkHit:
    video_id: str
    title: str
    start: float
    text: str
    score: float


@dataclass
class Citation:
    video_id: str
    title: str
    start: float
    url: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]


@dataclass
class RunSummary:
    new: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
```

- [ ] **Step 2: Write the failing test**

`tests/test_db.py`:
```python
from ytkb.db import connect, upsert_video, set_state, videos_by_state, count_by_state, VideoState, insert_chunks
from ytkb.models import VideoMeta, Chunk


def make_meta(vid="v1"):
    return VideoMeta(video_id=vid, title="Title " + vid, duration=600, upload_date="20240101", url=f"https://youtu.be/{vid}")


def test_upsert_and_state_transitions(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    row = videos_by_state(conn, [VideoState.DISCOVERED])
    assert len(row) == 1 and row[0]["video_id"] == "v1"

    set_state(conn, "v1", VideoState.TRANSCRIPT_FETCHED)
    assert videos_by_state(conn, [VideoState.DISCOVERED]) == []
    assert len(videos_by_state(conn, [VideoState.TRANSCRIPT_FETCHED])) == 1


def test_upsert_does_not_reset_state(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    set_state(conn, "v1", VideoState.INDEXED)
    upsert_video(conn, make_meta("v1"))  # re-listed on a later sync
    assert len(videos_by_state(conn, [VideoState.INDEXED])) == 1


def test_set_state_records_error_and_attempts(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    set_state(conn, "v1", VideoState.FAILED_FETCH, error="boom")
    set_state(conn, "v1", VideoState.FAILED_FETCH, error="boom2")
    v = videos_by_state(conn, [VideoState.FAILED_FETCH])[0]
    assert v["attempt_count"] == 2
    assert v["last_error"] == "boom2"


def test_count_by_state(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    upsert_video(conn, make_meta("v2"))
    set_state(conn, "v2", VideoState.INDEXED)
    counts = count_by_state(conn)
    assert counts[VideoState.DISCOVERED.value] == 1
    assert counts[VideoState.INDEXED.value] == 1


def test_insert_chunks_and_fts(tmp_path):
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    chunks = [Chunk(video_id="v1", idx=0, start=12.0, text="how to find a cofounder")]
    insert_chunks(conn, chunks, title_of={"v1": "Title v1"})
    rows = conn.execute(
        "SELECT c.video_id, c.start, c.text FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid WHERE chunks_fts MATCH ?",
        ("cofounder",),
    ).fetchall()
    assert rows[0]["video_id"] == "v1"
    assert rows[0]["start"] == 12.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ytkb.db'`

- [ ] **Step 4: Implement `ytkb/db.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add ytkb/models.py ytkb/db.py tests/test_db.py
git commit -m "feat: SQLite schema, video state machine, and FTS5 chunk store"
```

---

### Task 3: Channel resolution and video listing

**Files:**
- Create: `ytkb/channel.py`
- Test: `tests/test_channel.py`

**Interfaces:**
- Consumes: `ytkb.models.VideoMeta`.
- Produces:
  - `@dataclass ChannelInfo(channel_id: str, handle: str, title: str, uploads_url: str)`
  - `@dataclass ChannelFilters(max: int | None = None, since: str | None = None, min_length: int | None = None, no_shorts: bool = False, whisper: bool = True)` (`since` is `YYYYMMDD`).
  - `resolve_channel(url: str, *, ydl=None) -> ChannelInfo`
  - `list_videos(uploads_url: str, filters: ChannelFilters, *, ydl=None) -> list[VideoMeta]`
  - `_passes_filters(meta: VideoMeta, filters: ChannelFilters) -> bool` (helper; pure, unit-tested directly).
- The `ydl` parameter is an injectable object exposing `extract_info(url, download=False)` so tests pass a fake instead of hitting the network.

- [ ] **Step 1: Write the failing test (filters are pure — test them directly)**

`tests/test_channel.py`:
```python
from ytkb.channel import _passes_filters, ChannelFilters, list_videos, resolve_channel
from ytkb.models import VideoMeta


def vm(vid, dur=600, date="20240115", title="t"):
    return VideoMeta(video_id=vid, title=title, duration=dur, upload_date=date, url=f"https://youtu.be/{vid}")


def test_filters_no_shorts():
    f = ChannelFilters(no_shorts=True)
    assert _passes_filters(vm("a", dur=600), f) is True
    assert _passes_filters(vm("b", dur=30), f) is False


def test_filters_min_length_and_since():
    f = ChannelFilters(min_length=300, since="20240101")
    assert _passes_filters(vm("a", dur=600, date="20240115"), f) is True
    assert _passes_filters(vm("b", dur=100, date="20240115"), f) is False  # too short
    assert _passes_filters(vm("c", dur=600, date="20231201"), f) is False  # too old


class FakeYDL:
    def __init__(self, info):
        self._info = info

    def extract_info(self, url, download=False):
        return self._info


def test_resolve_channel_from_video_url():
    info = {"channel_id": "UC123", "uploader_id": "@yc", "channel": "Y Combinator"}
    ci = resolve_channel("https://youtu.be/abc", ydl=FakeYDL(info))
    assert ci.channel_id == "UC123"
    assert ci.handle == "@yc"
    assert ci.uploads_url == "https://www.youtube.com/channel/UC123/videos"


def test_list_videos_applies_max_and_filters():
    entries = {
        "entries": [
            {"id": "a", "title": "Long", "duration": 600, "upload_date": "20240115"},
            {"id": "b", "title": "Short", "duration": 30, "upload_date": "20240115"},
            {"id": "c", "title": "Long2", "duration": 700, "upload_date": "20240110"},
        ]
    }
    vids = list_videos("u", ChannelFilters(no_shorts=True, max=1), ydl=FakeYDL(entries))
    assert [v.video_id for v in vids] == ["a"]  # short filtered, max=1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_channel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ytkb.channel'`

- [ ] **Step 3: Implement `ytkb/channel.py`**

```python
from contextlib import contextmanager
from dataclasses import dataclass

from yt_dlp import YoutubeDL

from .models import VideoMeta


@dataclass
class ChannelInfo:
    channel_id: str
    handle: str
    title: str
    uploads_url: str


@dataclass
class ChannelFilters:
    max: int | None = None
    since: str | None = None      # YYYYMMDD
    min_length: int | None = None
    no_shorts: bool = False
    whisper: bool = True


@contextmanager
def _default_ydl(extra: dict | None = None):
    opts = {"quiet": True, "skip_download": True, "extract_flat": "in_playlist"}
    if extra:
        opts.update(extra)
    with YoutubeDL(opts) as ydl:
        yield ydl


def resolve_channel(url: str, *, ydl=None) -> ChannelInfo:
    if ydl is None:
        with _default_ydl() as real:
            info = real.extract_info(url, download=False)
    else:
        info = ydl.extract_info(url, download=False)
    channel_id = info.get("channel_id") or info.get("uploader_id")
    title = info.get("channel") or info.get("uploader") or channel_id
    handle = info.get("uploader_id") or info.get("channel_id") or ""
    return ChannelInfo(
        channel_id=channel_id,
        handle=handle,
        title=title,
        uploads_url=f"https://www.youtube.com/channel/{channel_id}/videos",
    )


def _passes_filters(meta: VideoMeta, f: ChannelFilters) -> bool:
    dur = meta.duration or 0
    if f.no_shorts and dur and dur < 60:
        return False
    if f.min_length and dur and dur < f.min_length:
        return False
    if f.since and meta.upload_date and meta.upload_date < f.since:
        return False
    return True


def list_videos(uploads_url: str, filters: ChannelFilters, *, ydl=None) -> list[VideoMeta]:
    if ydl is None:
        with _default_ydl() as real:
            info = real.extract_info(uploads_url, download=False)
    else:
        info = ydl.extract_info(uploads_url, download=False)
    out: list[VideoMeta] = []
    for e in info.get("entries", []) or []:
        meta = VideoMeta(
            video_id=e["id"],
            title=e.get("title") or e["id"],
            duration=e.get("duration"),
            upload_date=e.get("upload_date"),
            url=f"https://youtu.be/{e['id']}",
        )
        if _passes_filters(meta, filters):
            out.append(meta)
        if filters.max and len(out) >= filters.max:
            break
    return out
```

> Note: flat playlist extraction may omit `duration`/`upload_date`. When `no_shorts`/`min_length`/`since` filters are set and these fields are missing, the pipeline (Task 10) fills them by fetching full metadata per video before deciding; `_passes_filters` treats missing duration as "keep" so nothing is silently dropped.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_channel.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/channel.py tests/test_channel.py
git commit -m "feat: channel resolution and filtered video listing via yt-dlp"
```

---

### Task 4: Caption fetching and raw transcript storage

**Files:**
- Create: `ytkb/transcripts.py`
- Test: `tests/test_transcripts.py`

**Interfaces:**
- Consumes: `ytkb.models.Segment`.
- Produces:
  - `parse_json3(data: dict) -> list[Segment]` (parses YouTube json3 caption payload; pure).
  - `fetch_captions(video_id: str, languages: list[str], *, downloader=None) -> list[Segment] | None` (returns `None` if no captions). `downloader(video_id, languages) -> dict | None` is injectable and returns the raw json3 dict.
  - `save_raw(path: Path, segments: list[Segment]) -> None`
  - `load_raw(path: Path) -> list[Segment]`

- [ ] **Step 1: Write the failing test**

`tests/test_transcripts.py`:
```python
from ytkb.transcripts import parse_json3, fetch_captions, save_raw, load_raw
from ytkb.models import Segment

JSON3 = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "hello "}, {"utf8": "world"}]},
        {"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "next line"}]},
        {"tStartMs": 4000, "segs": [{"utf8": "\n"}]},  # whitespace-only -> dropped
    ]
}


def test_parse_json3():
    segs = parse_json3(JSON3)
    assert segs == [
        Segment(start=0.0, end=2.0, text="hello world"),
        Segment(start=2.0, end=3.5, text="next line"),
    ]


def test_fetch_captions_returns_none_when_missing():
    assert fetch_captions("v1", ["en"], downloader=lambda vid, langs: None) is None


def test_fetch_captions_parses():
    segs = fetch_captions("v1", ["en"], downloader=lambda vid, langs: JSON3)
    assert segs[0].text == "hello world"


def test_save_and_load_roundtrip(tmp_path):
    segs = [Segment(0.0, 2.0, "a"), Segment(2.0, 3.0, "b")]
    p = tmp_path / "v1.raw.json"
    save_raw(p, segs)
    assert load_raw(p) == segs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transcripts.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/transcripts.py`**

```python
import json
from pathlib import Path

from yt_dlp import YoutubeDL

from .models import Segment


def parse_json3(data: dict) -> list[Segment]:
    out: list[Segment] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        start = ev["tStartMs"] / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        out.append(Segment(start=start, end=start + dur, text=text))
    return out


def _yt_dlp_download(video_id: str, languages: list[str]) -> dict | None:
    opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "json3",
        "subtitleslangs": languages,
    }
    url = f"https://youtu.be/{video_id}"
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    subs = {**(info.get("subtitles") or {}), **(info.get("automatic_captions") or {})}
    for lang in languages:
        tracks = subs.get(lang) or subs.get(lang.split("-")[0])
        if not tracks:
            continue
        for t in tracks:
            if t.get("ext") == "json3":
                data = ydl.urlopen(t["url"]).read()
                return json.loads(data)
    return None


def fetch_captions(video_id: str, languages: list[str], *, downloader=None) -> list[Segment] | None:
    downloader = downloader or _yt_dlp_download
    raw = downloader(video_id, languages)
    if raw is None:
        return None
    segs = parse_json3(raw)
    return segs or None


def save_raw(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.__dict__ for s in segments], ensure_ascii=False))


def load_raw(path: Path) -> list[Segment]:
    data = json.loads(Path(path).read_text())
    return [Segment(**d) for d in data]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcripts.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/transcripts.py tests/test_transcripts.py
git commit -m "feat: caption fetching (json3) and raw transcript storage"
```

---

### Task 5: Whisper fallback transcription

**Files:**
- Create: `ytkb/whisper_fallback.py`
- Test: `tests/test_whisper_fallback.py`

**Interfaces:**
- Consumes: `ytkb.models.Segment`.
- Produces: `transcribe(video_id: str, model_size: str, *, audio_path_fn=None, model=None) -> list[Segment]`. `audio_path_fn(video_id) -> Path` downloads audio and returns its path; `model` is an object exposing `.transcribe(path) -> (iterable_of_segments, info)` where each segment has `.start`, `.end`, `.text` (matches faster-whisper's `WhisperModel`). Both injectable for tests.

- [ ] **Step 1: Write the failing test (fake model + fake downloader — no real audio/model)**

`tests/test_whisper_fallback.py`:
```python
from types import SimpleNamespace
from ytkb.whisper_fallback import transcribe


class FakeModel:
    def transcribe(self, path):
        segs = [
            SimpleNamespace(start=0.0, end=1.0, text="hello"),
            SimpleNamespace(start=1.0, end=2.0, text=" world"),
        ]
        return segs, SimpleNamespace(language="en")


def test_transcribe_maps_segments(tmp_path):
    audio = tmp_path / "v1.m4a"
    audio.write_bytes(b"x")
    out = transcribe("v1", "base", audio_path_fn=lambda vid: audio, model=FakeModel())
    assert [(s.start, s.text) for s in out] == [(0.0, "hello"), (1.0, "world")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_whisper_fallback.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/whisper_fallback.py`**

```python
import tempfile
from pathlib import Path

from yt_dlp import YoutubeDL

from .models import Segment


def _download_audio(video_id: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    out = tmp / f"{video_id}.%(ext)s"
    opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": str(out),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([f"https://youtu.be/{video_id}"])
    return next(tmp.glob(f"{video_id}.*"))


def transcribe(video_id: str, model_size: str, *, audio_path_fn=None, model=None) -> list[Segment]:
    audio_path_fn = audio_path_fn or _download_audio
    if model is None:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="auto", compute_type="int8")
    path = audio_path_fn(video_id)
    segments, _info = model.transcribe(str(path))
    return [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_whisper_fallback.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ytkb/whisper_fallback.py tests/test_whisper_fallback.py
git commit -m "feat: local faster-whisper fallback transcription"
```

---

### Task 6: SponsorBlock client and segment stripping

**Files:**
- Create: `ytkb/sponsorblock.py`
- Test: `tests/test_sponsorblock.py`

**Interfaces:**
- Consumes: `ytkb.models.Segment`.
- Produces:
  - `get_segments(video_id: str, *, http_get=None) -> list[tuple[float, float]]` — returns sponsor/selfpromo `(start, end)` ranges; `[]` if none. `http_get(url, params) -> list[dict] | None` is injectable; on HTTP 404 (no data) returns `[]`.
  - `strip_segments(segments: list[Segment], ad_ranges: list[tuple[float, float]]) -> list[Segment]` — drops any segment whose midpoint falls inside an ad range (pure).

- [ ] **Step 1: Write the failing test**

`tests/test_sponsorblock.py`:
```python
from ytkb.sponsorblock import get_segments, strip_segments
from ytkb.models import Segment


def test_strip_segments_drops_overlapping():
    segs = [
        Segment(0, 5, "intro"),
        Segment(5, 10, "sponsor read"),   # midpoint 7.5 in [5,12]
        Segment(12, 20, "content"),
    ]
    out = strip_segments(segs, [(5.0, 12.0)])
    assert [s.text for s in out] == ["intro", "content"]


def test_get_segments_parses_categories():
    api = [
        {"segment": [30.0, 45.0], "category": "sponsor"},
        {"segment": [60.0, 65.0], "category": "interaction"},  # ignored
        {"segment": [90.0, 95.0], "category": "selfpromo"},
    ]
    ranges = get_segments("v1", http_get=lambda url, params: api)
    assert ranges == [(30.0, 45.0), (90.0, 95.0)]


def test_get_segments_handles_no_data():
    assert get_segments("v1", http_get=lambda url, params: None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sponsorblock.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/sponsorblock.py`**

```python
import requests

from .models import Segment

API = "https://sponsor.ajax.sb.org/api/skipSegments"  # mirror of sponsor.ajax... ; see note
CATEGORIES = ("sponsor", "selfpromo")


def _http_get(url: str, params: dict):
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_segments(video_id: str, *, http_get=None) -> list[tuple[float, float]]:
    http_get = http_get or _http_get
    data = http_get(
        "https://sponsor.ajax.sb.org/api/skipSegments",
        {"videoID": video_id, "categories": '["sponsor","selfpromo"]'},
    )
    if not data:
        return []
    out = []
    for item in data:
        if item.get("category") in CATEGORIES:
            s, e = item["segment"]
            out.append((float(s), float(e)))
    return out


def strip_segments(segments: list[Segment], ad_ranges: list[tuple[float, float]]) -> list[Segment]:
    def in_ad(seg: Segment) -> bool:
        mid = (seg.start + seg.end) / 2
        return any(lo <= mid <= hi for lo, hi in ad_ranges)

    return [s for s in segments if not in_ad(s)]
```

> Note: the canonical SponsorBlock endpoint is `https://sponsor.ajax.sb.org/api/skipSegments` (privacy mirror) or `https://api.sponsor.ajay.app/api/skipSegments`. Use `https://api.sponsor.ajay.app/api/skipSegments` as the real default; the test injects `http_get` so the URL constant is not exercised in tests. Update the two literals to that URL during implementation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sponsorblock.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/sponsorblock.py tests/test_sponsorblock.py
git commit -m "feat: SponsorBlock client and segment stripping"
```

---

### Task 7: OpenRouter LLM client and ad-strip orchestration

**Files:**
- Create: `ytkb/llm.py`, `ytkb/adstrip.py`
- Test: `tests/test_llm.py`, `tests/test_adstrip.py`

**Interfaces:**
- Consumes: `ytkb.config.Config`, `ytkb.models.Segment`, `ytkb.sponsorblock`.
- Produces:
  - `ytkb.llm.LLMClient(api_key: str | None, base_url: str = "https://openrouter.ai/api/v1", *, client=None)` with:
    - `complete(messages: list[dict], model: str) -> str`
    - `chat_with_tools(messages: list[dict], model: str, tools: list[dict]) -> Any` (returns the raw choice message; used by the agent).
    - `client` is an injectable object matching the `openai.OpenAI` chat interface for tests.
  - `ytkb.adstrip.strip_ads(video_id: str, segments: list[Segment], llm: LLMClient, cfg: Config, *, sb_get=None) -> tuple[list[Segment], str]` — returns `(clean_segments, method)` where method is `"sponsorblock"`, `"llm"`, or `"none"`. `sb_get` injects `sponsorblock.get_segments`.

- [ ] **Step 1: Write the failing test for LLMClient**

`tests/test_llm.py`:
```python
from types import SimpleNamespace
from ytkb.llm import LLMClient


class FakeChat:
    def __init__(self, content):
        self._content = content
        self.calls = []

    class _Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, model, messages, **kwargs):
            self.outer.calls.append({"model": model, "messages": messages, **kwargs})
            msg = SimpleNamespace(content=self.outer._content, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    @property
    def completions(self):
        return FakeChat._Completions(self)


class FakeClient:
    def __init__(self, content):
        self.chat = FakeChat(content)


def test_complete_returns_text():
    llm = LLMClient(api_key="k", client=FakeClient("hi there"))
    out = llm.complete([{"role": "user", "content": "x"}], model="m")
    assert out == "hi there"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/llm.py`**

```python
from typing import Any


class LLMClient:
    def __init__(self, api_key: str | None, base_url: str = "https://openrouter.ai/api/v1", *, client=None):
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[dict], model: str) -> str:
        resp = self._client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    def chat_with_tools(self, messages: list[dict], model: str, tools: list[dict]) -> Any:
        resp = self._client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto"
        )
        return resp.choices[0].message
```

- [ ] **Step 4: Write the failing test for ad-strip orchestration**

`tests/test_adstrip.py`:
```python
from ytkb.adstrip import strip_ads
from ytkb.models import Segment
from ytkb.config import load_config


def segs():
    return [Segment(0, 5, "intro"), Segment(5, 10, "sponsor"), Segment(12, 20, "content")]


def test_sponsorblock_path(tmp_path):
    cfg = load_config(tmp_path)
    clean, method = strip_ads("v1", segs(), llm=None, cfg=cfg, sb_get=lambda vid: [(5.0, 11.0)])
    assert method == "sponsorblock"
    assert [s.text for s in clean] == ["intro", "content"]


class FakeLLM:
    def complete(self, messages, model):
        # returns the indices (0-based) of ad segments as CSV
        return "1"


def test_llm_fallback_when_no_sponsorblock(tmp_path):
    cfg = load_config(tmp_path)
    clean, method = strip_ads("v1", segs(), llm=FakeLLM(), cfg=cfg, sb_get=lambda vid: [])
    assert method == "llm"
    assert [s.text for s in clean] == ["intro", "content"]


def test_no_ads_returns_all(tmp_path):
    cfg = load_config(tmp_path)

    class NoAdLLM:
        def complete(self, messages, model):
            return ""

    clean, method = strip_ads("v1", segs(), llm=NoAdLLM(), cfg=cfg, sb_get=lambda vid: [])
    assert method == "none"
    assert len(clean) == 3
```

- [ ] **Step 5: Run test to verify it fails**

Run: `uv run pytest tests/test_adstrip.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 6: Implement `ytkb/adstrip.py`**

```python
from .config import Config
from .models import Segment
from .sponsorblock import get_segments as sb_get_segments, strip_segments

PROMPT = (
    "You are removing in-video sponsor/advertising reads from a transcript. "
    "Below are numbered transcript segments. Return ONLY a comma-separated list of the "
    "segment numbers (0-based) that are sponsor reads, ads, or self-promotion. "
    "If there are none, return an empty string.\n\n"
)


def _llm_ad_indices(segments: list[Segment], llm, model: str) -> set[int]:
    numbered = "\n".join(f"{i}: {s.text}" for i, s in enumerate(segments))
    raw = llm.complete([{"role": "user", "content": PROMPT + numbered}], model=model)
    idxs: set[int] = set()
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit():
            idxs.add(int(tok))
    return idxs


def strip_ads(video_id, segments, llm, cfg: Config, *, sb_get=None) -> tuple[list[Segment], str]:
    sb_get = sb_get or sb_get_segments
    ad_ranges = sb_get(video_id)
    if ad_ranges:
        return strip_segments(segments, ad_ranges), "sponsorblock"
    if llm is None:
        return list(segments), "none"
    idxs = _llm_ad_indices(segments, llm, cfg.adstrip_model)
    if not idxs:
        return list(segments), "none"
    return [s for i, s in enumerate(segments) if i not in idxs], "llm"
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py tests/test_adstrip.py -v`
Expected: PASS (4 tests)

- [ ] **Step 8: Commit**

```bash
git add ytkb/llm.py ytkb/adstrip.py tests/test_llm.py tests/test_adstrip.py
git commit -m "feat: OpenRouter LLM client and SponsorBlock->LLM ad stripping"
```

---

### Task 8: Transcript chunking

**Files:**
- Create: `ytkb/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `ytkb.models.Segment`, `ytkb.models.Chunk`.
- Produces: `chunk_segments(video_id: str, segments: list[Segment], chunk_tokens: int, overlap: int) -> list[Chunk]`. Tokens are approximated as whitespace words; each chunk's `start` is the start time of its first segment; chunks carry `idx` starting at 0; overlap is in words.

- [ ] **Step 1: Write the failing test**

`tests/test_chunking.py`:
```python
from ytkb.chunking import chunk_segments
from ytkb.models import Segment


def test_chunks_respect_token_budget_and_timestamps():
    segs = [Segment(start=float(i * 5), end=float(i * 5 + 5), text="word " * 10) for i in range(5)]
    chunks = chunk_segments("v1", segs, chunk_tokens=20, overlap=0)
    assert chunks[0].idx == 0
    assert chunks[0].start == 0.0
    assert all(len(c.text.split()) <= 20 for c in chunks)
    # 50 words / 20 per chunk -> 3 chunks
    assert len(chunks) == 3


def test_overlap_repeats_words():
    segs = [Segment(0, 5, "a b c d e f g h")]
    chunks = chunk_segments("v1", segs, chunk_tokens=4, overlap=2)
    assert chunks[0].text.split() == ["a", "b", "c", "d"]
    assert chunks[1].text.split()[:2] == ["c", "d"]


def test_empty_segments():
    assert chunk_segments("v1", [], 100, 0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/chunking.py`**

```python
from .models import Segment, Chunk


def chunk_segments(video_id: str, segments: list[Segment], chunk_tokens: int, overlap: int) -> list[Chunk]:
    # Build a flat list of (word, start_time) so each chunk knows its first word's timestamp.
    words: list[tuple[str, float]] = []
    for seg in segments:
        for w in seg.text.split():
            words.append((w, seg.start))
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_tokens - overlap)
    idx = 0
    i = 0
    while i < len(words):
        window = words[i : i + chunk_tokens]
        text = " ".join(w for w, _ in window)
        chunks.append(Chunk(video_id=video_id, idx=idx, start=window[0][1], text=text))
        idx += 1
        if i + chunk_tokens >= len(words):
            break
        i += step
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/chunking.py tests/test_chunking.py
git commit -m "feat: timestamp-preserving transcript chunking"
```

---

### Task 9: Embeddings and the channel store (LanceDB + FTS5 search)

**Files:**
- Create: `ytkb/embeddings.py`, `ytkb/store.py`
- Test: `tests/test_embeddings.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `ytkb.config.Config`, `ytkb.paths.ChannelPaths`, `ytkb.db`, `ytkb.models` (`Chunk`, `ChunkHit`, `VideoMeta`).
- Produces:
  - `ytkb.embeddings.Embedder(model_name: str, *, backend=None)` with `embed(texts: list[str]) -> list[list[float]]` and property `dim: int`. `backend` injectable; default uses `fastembed.TextEmbedding`.
  - `ytkb.store.ChannelStore(paths: ChannelPaths, conn, embedder: Embedder)` with:
    - `add(chunks: list[Chunk], title_of: dict[str, str]) -> None` (embeds, writes LanceDB table `chunks`, and calls `db.insert_chunks`).
    - `keyword_search(query: str, k: int) -> list[ChunkHit]` (FTS5 BM25).
    - `semantic_search(query: str, k: int) -> list[ChunkHit]` (LanceDB vector search).
    - `read_around(video_id: str, around_ts: float | None, window: float) -> str` (returns clean text near a timestamp).
    - `list_videos(contains: str | None = None) -> list[dict]` (title + id + upload_date from `videos`).

- [ ] **Step 1: Write the failing test for Embedder (fake backend)**

`tests/test_embeddings.py`:
```python
from ytkb.embeddings import Embedder


class FakeBackend:
    def embed(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]


def test_embed_uses_backend():
    emb = Embedder("fake", backend=FakeBackend())
    out = emb.embed(["ab", "abc"])
    assert out == [[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
    assert emb.dim == 3
```

- [ ] **Step 2: Run test to verify it fails, then implement `ytkb/embeddings.py`**

Run: `uv run pytest tests/test_embeddings.py -v` → FAIL (`ModuleNotFoundError`).

```python
class Embedder:
    def __init__(self, model_name: str, *, backend=None):
        self.model_name = model_name
        self._backend = backend
        self._dim: int | None = None

    def _ensure(self):
        if self._backend is None:
            from fastembed import TextEmbedding
            self._backend = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        vectors = [list(map(float, v)) for v in self._backend.embed(texts)]
        if vectors:
            self._dim = len(vectors[0])
        return vectors

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["x"])[0])
        return self._dim
```

Run again → PASS.

- [ ] **Step 3: Write the failing test for ChannelStore**

`tests/test_store.py`:
```python
from ytkb.paths import ChannelPaths
from ytkb.db import connect, upsert_video
from ytkb.embeddings import Embedder
from ytkb.store import ChannelStore
from ytkb.models import VideoMeta, Chunk


class HashBackend:
    """Deterministic 8-dim embedding from word hashes so semantic search is testable."""
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 8
            for w in t.lower().split():
                v[hash(w) % 8] += 1.0
            out.append(v)
        return out


def build_store(tmp_path):
    paths = ChannelPaths.for_slug(tmp_path, "c")
    paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v1", "Cofounders", 600, "20240101", "https://youtu.be/v1"))
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    chunks = [
        Chunk("v1", 0, 10.0, "how to find a cofounder for your startup"),
        Chunk("v1", 1, 40.0, "pricing your product and revenue models"),
    ]
    store.add(chunks, title_of={"v1": "Cofounders"})
    return store


def test_keyword_search_finds_exact(tmp_path):
    store = build_store(tmp_path)
    hits = store.keyword_search("cofounder", k=5)
    assert hits[0].video_id == "v1"
    assert hits[0].start == 10.0


def test_semantic_search_returns_hits(tmp_path):
    store = build_store(tmp_path)
    hits = store.semantic_search("cofounder", k=1)
    assert len(hits) == 1
    assert hits[0].title == "Cofounders"


def test_list_videos(tmp_path):
    store = build_store(tmp_path)
    vids = store.list_videos()
    assert vids[0]["title"] == "Cofounders"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ytkb.store'`

- [ ] **Step 5: Implement `ytkb/store.py`**

```python
import lancedb

from .db import insert_chunks
from .embeddings import Embedder
from .models import Chunk, ChunkHit
from .paths import ChannelPaths
from .transcripts import load_raw

TABLE = "chunks"


class ChannelStore:
    def __init__(self, paths: ChannelPaths, conn, embedder: Embedder):
        self.paths = paths
        self.conn = conn
        self.embedder = embedder
        self._db = lancedb.connect(str(paths.vectors_dir))

    def _table(self):
        if TABLE in self._db.table_names():
            return self._db.open_table(TABLE)
        return None

    def add(self, chunks: list[Chunk], title_of: dict[str, str]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        rows = [
            {"vector": vectors[i], "video_id": c.video_id, "start": c.start,
             "title": title_of[c.video_id], "text": c.text}
            for i, c in enumerate(chunks)
        ]
        table = self._table()
        if table is None:
            self._db.create_table(TABLE, data=rows)
        else:
            table.add(rows)
        insert_chunks(self.conn, chunks, title_of)

    def keyword_search(self, query: str, k: int) -> list[ChunkHit]:
        rows = self.conn.execute(
            """SELECT c.video_id, c.title, c.start, c.text, bm25(chunks_fts) AS score
               FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid
               WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?""",
            (query, k),
        ).fetchall()
        return [ChunkHit(r["video_id"], r["title"], r["start"], r["text"], r["score"]) for r in rows]

    def semantic_search(self, query: str, k: int) -> list[ChunkHit]:
        table = self._table()
        if table is None:
            return []
        qv = self.embedder.embed([query])[0]
        res = table.search(qv).limit(k).to_list()
        return [
            ChunkHit(r["video_id"], r["title"], r["start"], r["text"], float(r.get("_distance", 0.0)))
            for r in res
        ]

    def read_around(self, video_id: str, around_ts: float | None, window: float = 90.0) -> str:
        segs = load_raw(self.paths.clean_segments_path(video_id)) if False else None
        text = self.paths.clean_path(video_id).read_text()
        if around_ts is None:
            return text[:4000]
        # clean.txt stores plain text; for windowing we fall back to chunk rows near the ts.
        rows = self.conn.execute(
            "SELECT text FROM chunks WHERE video_id=? AND start BETWEEN ? AND ? ORDER BY start",
            (video_id, around_ts - window, around_ts + window),
        ).fetchall()
        return "\n".join(r["text"] for r in rows) or text[:4000]

    def list_videos(self, contains: str | None = None) -> list[dict]:
        if contains:
            rows = self.conn.execute(
                "SELECT video_id, title, upload_date FROM videos WHERE title LIKE ? AND state='indexed'",
                (f"%{contains}%",),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT video_id, title, upload_date FROM videos WHERE state='indexed'"
            ).fetchall()
        return [dict(r) for r in rows]
```

> Implementation note: remove the dead `load_raw(... if False ...)` line — it is shown only to flag that `read_around` intentionally reads from chunk rows (which carry timestamps) rather than the flat `clean.txt`. Final `read_around` should be just the chunk-row query with the `clean.txt` fallback.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add ytkb/embeddings.py ytkb/store.py tests/test_embeddings.py tests/test_store.py
git commit -m "feat: local embeddings and channel store (LanceDB + FTS5 search)"
```

---

### Task 10: Per-video pipeline orchestration

**Files:**
- Create: `ytkb/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass ChannelContext(paths, conn, cfg, llm, store, filters, whisper_enabled)`.
  - `process_video(ctx: ChannelContext, meta: VideoMeta, *, fetch=None, whisper=None, strip=None) -> VideoState` — runs the full pipeline for one video, writing files, updating state at each step, and writing `clean.txt`. Returns the final `VideoState`. `fetch`, `whisper`, and `strip` are injectable callables defaulting to `transcripts.fetch_captions`, `whisper_fallback.transcribe`, and `adstrip.strip_ads`.

- [ ] **Step 1: Write the failing test (all I/O injected; no network)**

`tests/test_pipeline.py`:
```python
from ytkb.paths import ChannelPaths
from ytkb.db import connect, upsert_video, get_video, VideoState
from ytkb.embeddings import Embedder
from ytkb.store import ChannelStore
from ytkb.channel import ChannelFilters
from ytkb.pipeline import ChannelContext, process_video
from ytkb.models import VideoMeta, Segment
from ytkb.config import load_config
from tests.test_store import HashBackend


def make_ctx(tmp_path, whisper_enabled=True):
    paths = ChannelPaths.for_slug(tmp_path, "c")
    paths.ensure()
    conn = connect(paths.db)
    cfg = load_config(tmp_path)
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    return ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=None, store=store,
                          filters=ChannelFilters(), whisper_enabled=whisper_enabled)


def meta(vid="v1"):
    return VideoMeta(vid, "Title", 600, "20240101", f"https://youtu.be/{vid}")


def test_pipeline_happy_path(tmp_path):
    ctx = make_ctx(tmp_path)
    upsert_video(ctx.conn, meta())
    segs = [Segment(0, 5, "find a cofounder"), Segment(5, 10, "this is sponsored by acme")]
    state = process_video(
        ctx, meta(),
        fetch=lambda vid, langs: segs,
        strip=lambda vid, s, llm, cfg, **k: ([segs[0]], "llm"),
    )
    assert state == VideoState.INDEXED
    assert get_video(ctx.conn, "v1")["state"] == "indexed"
    assert ctx.paths.clean_path("v1").exists()
    assert ctx.store.keyword_search("cofounder", 5)[0].video_id == "v1"


def test_pipeline_no_captions_whisper_disabled(tmp_path):
    ctx = make_ctx(tmp_path, whisper_enabled=False)
    upsert_video(ctx.conn, meta())
    state = process_video(ctx, meta(), fetch=lambda vid, langs: None)
    assert state == VideoState.NO_TRANSCRIPT


def test_pipeline_whisper_fallback(tmp_path):
    ctx = make_ctx(tmp_path, whisper_enabled=True)
    upsert_video(ctx.conn, meta())
    segs = [Segment(0, 5, "whisper text about cofounder")]
    state = process_video(
        ctx, meta(),
        fetch=lambda vid, langs: None,
        whisper=lambda vid, model: segs,
        strip=lambda vid, s, llm, cfg, **k: (s, "none"),
    )
    assert state == VideoState.INDEXED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/pipeline.py`**

```python
from dataclasses import dataclass

from . import transcripts, whisper_fallback, adstrip
from .channel import ChannelFilters
from .chunking import chunk_segments
from .config import Config
from .db import set_state, set_ad_method, VideoState
from .models import VideoMeta
from .store import ChannelStore


@dataclass
class ChannelContext:
    paths: object
    conn: object
    cfg: Config
    llm: object
    store: ChannelStore
    filters: ChannelFilters
    whisper_enabled: bool


def process_video(ctx: ChannelContext, meta: VideoMeta, *, fetch=None, whisper=None, strip=None) -> VideoState:
    fetch = fetch or transcripts.fetch_captions
    whisper = whisper or whisper_fallback.transcribe
    strip = strip or adstrip.strip_ads
    vid = meta.video_id

    # 1. transcript
    try:
        segments = fetch(vid, ctx.cfg.languages)
    except Exception as e:  # network / extraction failure -> retryable
        set_state(ctx.conn, vid, VideoState.FAILED_FETCH, error=str(e))
        return VideoState.FAILED_FETCH

    if segments:
        set_state(ctx.conn, vid, VideoState.TRANSCRIPT_FETCHED)
    elif ctx.whisper_enabled:
        try:
            segments = whisper(vid, ctx.cfg.whisper_model)
        except Exception as e:
            set_state(ctx.conn, vid, VideoState.FAILED_FETCH, error=f"whisper: {e}")
            return VideoState.FAILED_FETCH
        if not segments:
            set_state(ctx.conn, vid, VideoState.NO_TRANSCRIPT, error="whisper produced nothing")
            return VideoState.NO_TRANSCRIPT
        set_state(ctx.conn, vid, VideoState.WHISPER_TRANSCRIBED)
    else:
        set_state(ctx.conn, vid, VideoState.NO_TRANSCRIPT, error="no captions; whisper disabled")
        return VideoState.NO_TRANSCRIPT

    transcripts.save_raw(ctx.paths.raw_path(vid), segments)

    # 2. ad stripping
    clean_segments, method = strip(vid, segments, ctx.llm, ctx.cfg)
    set_ad_method(ctx.conn, vid, method)
    ctx.paths.clean_path(vid).write_text(
        " ".join(s.text for s in clean_segments), encoding="utf-8"
    )
    set_state(ctx.conn, vid, VideoState.AD_STRIPPED)

    # 3. chunk + index
    try:
        chunks = chunk_segments(vid, clean_segments, ctx.cfg.chunk_tokens, ctx.cfg.chunk_overlap)
        ctx.store.add(chunks, title_of={vid: meta.title})
    except Exception as e:
        set_state(ctx.conn, vid, VideoState.FAILED_EMBED, error=str(e))
        return VideoState.FAILED_EMBED

    set_state(ctx.conn, vid, VideoState.INDEXED)
    return VideoState.INDEXED
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/pipeline.py tests/test_pipeline.py
git commit -m "feat: resumable per-video ingestion pipeline with state transitions"
```

---

### Task 11: Sync, retry, status, and channel registry

**Files:**
- Create: `ytkb/sync.py`
- Test: `tests/test_sync.py`

**Interfaces:**
- Consumes: everything above.
- Produces (all take `cfg: Config`):
  - `save_channel(cfg, slug, info: ChannelInfo, filters: ChannelFilters) -> None` (writes `channel.json`).
  - `load_channel(cfg, slug) -> tuple[ChannelInfo, ChannelFilters]`.
  - `add_channel(cfg, url, filters, name=None, *, resolver=None) -> str` (resolves channel, picks slug, saves; returns slug). `resolver` injects `channel.resolve_channel`.
  - `build_context(cfg, slug) -> ChannelContext`.
  - `sync_channel(cfg, slug, *, dry_run=False, lister=None, process=None) -> RunSummary` (lists videos, upserts, processes videos in states `DISCOVERED`/`FAILED_FETCH`/`FAILED_EMBED`). `lister` injects `channel.list_videos`; `process` injects `pipeline.process_video`.
  - `retry_channel(cfg, slug, *, process=None) -> RunSummary` (re-processes `FAILED_FETCH`/`FAILED_EMBED`/`NO_TRANSCRIPT`).
  - `channel_status(cfg, slug) -> dict` (counts by state + last_sync).

- [ ] **Step 1: Write the failing test**

`tests/test_sync.py`:
```python
from ytkb.config import load_config
from ytkb.channel import ChannelInfo, ChannelFilters
from ytkb.db import VideoState, get_video
from ytkb.models import VideoMeta
from ytkb import sync


def info():
    return ChannelInfo("UC1", "@yc", "Y Combinator", "https://www.youtube.com/channel/UC1/videos")


def test_add_and_load_channel(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "https://youtu.be/x", ChannelFilters(no_shorts=True),
                            resolver=lambda url, **k: info())
    assert slug == "y-combinator"
    ci, filters = sync.load_channel(cfg, slug)
    assert ci.channel_id == "UC1"
    assert filters.no_shorts is True


def test_sync_processes_new_videos(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    vids = [VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a"),
            VideoMeta("b", "B", 600, "20240102", "https://youtu.be/b")]
    processed = []

    def fake_process(ctx, meta, **k):
        processed.append(meta.video_id)
        from ytkb.db import set_state
        set_state(ctx.conn, meta.video_id, VideoState.INDEXED)
        return VideoState.INDEXED

    summary = sync.sync_channel(cfg, slug, lister=lambda url, f, **k: vids, process=fake_process)
    assert sorted(processed) == ["a", "b"]
    assert summary.new == 2 and summary.done == 2


def test_sync_dry_run_processes_nothing(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    vids = [VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a")]
    called = []
    summary = sync.sync_channel(cfg, slug, dry_run=True, lister=lambda url, f, **k: vids,
                                process=lambda *a, **k: called.append(1))
    assert called == []
    assert summary.new == 1


def test_sync_skips_already_indexed(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    vids = [VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a")]

    def fake_process(ctx, meta, **k):
        from ytkb.db import set_state
        set_state(ctx.conn, meta.video_id, VideoState.INDEXED)
        return VideoState.INDEXED

    sync.sync_channel(cfg, slug, lister=lambda url, f, **k: vids, process=fake_process)
    count = []
    sync.sync_channel(cfg, slug, lister=lambda url, f, **k: vids,
                      process=lambda ctx, meta, **k: count.append(1))
    assert count == []  # already indexed -> not reprocessed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sync.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/sync.py`**

```python
import json
from dataclasses import asdict

from .channel import ChannelInfo, ChannelFilters, resolve_channel, list_videos
from .config import Config
from .db import connect, upsert_video, videos_by_state, count_by_state, record_run, VideoState
from .embeddings import Embedder
from .llm import LLMClient
from .models import RunSummary
from .paths import ChannelPaths, slugify, list_channel_slugs
from .pipeline import ChannelContext, process_video
from .store import ChannelStore

PROCESS_STATES = [VideoState.DISCOVERED, VideoState.FAILED_FETCH, VideoState.FAILED_EMBED]
RETRY_STATES = [VideoState.FAILED_FETCH, VideoState.FAILED_EMBED, VideoState.NO_TRANSCRIPT]


def save_channel(cfg: Config, slug: str, info: ChannelInfo, filters: ChannelFilters) -> None:
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    paths.ensure()
    paths.channel_json.write_text(json.dumps(
        {"info": asdict(info), "filters": asdict(filters)}, indent=2
    ))


def load_channel(cfg: Config, slug: str) -> tuple[ChannelInfo, ChannelFilters]:
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    data = json.loads(paths.channel_json.read_text())
    return ChannelInfo(**data["info"]), ChannelFilters(**data["filters"])


def add_channel(cfg, url, filters: ChannelFilters, name=None, *, resolver=None) -> str:
    resolver = resolver or resolve_channel
    info = resolver(url)
    slug = slugify(name or info.title or info.handle)
    save_channel(cfg, slug, info, filters)
    return slug


def build_context(cfg: Config, slug: str) -> ChannelContext:
    info, filters = load_channel(cfg, slug)
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    paths.ensure()
    conn = connect(paths.db)
    embedder = Embedder(cfg.embedding_model)
    store = ChannelStore(paths, conn, embedder)
    llm = LLMClient(cfg.openrouter_api_key) if cfg.openrouter_api_key else None
    return ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=llm, store=store,
                          filters=filters, whisper_enabled=filters.whisper)


def _summarize(states: list[VideoState]) -> RunSummary:
    s = RunSummary()
    for st in states:
        if st == VideoState.INDEXED:
            s.done += 1
        elif st in (VideoState.NO_TRANSCRIPT,):
            s.skipped += 1
        else:
            s.failed += 1
    return s


def sync_channel(cfg, slug, *, dry_run=False, lister=None, process=None) -> RunSummary:
    lister = lister or list_videos
    process = process or process_video
    info, filters = load_channel(cfg, slug)
    ctx = build_context(cfg, slug)

    videos = lister(info.uploads_url, filters)
    summary = RunSummary(new=0)
    new_ids = []
    for meta in videos:
        if ctx.conn.execute("SELECT 1 FROM videos WHERE video_id=?", (meta.video_id,)).fetchone() is None:
            summary.new += 1
            new_ids.append(meta.video_id)
        upsert_video(ctx.conn, meta)

    if dry_run:
        record_run(ctx.conn, summary, "sync-dry")
        return summary

    todo = videos_by_state(ctx.conn, PROCESS_STATES)
    by_id = {m.video_id: m for m in videos}
    results = []
    for row in todo:
        meta = by_id.get(row["video_id"])
        if meta is None:
            continue
        results.append(process(ctx, meta))
    s = _summarize(results)
    s.new = summary.new
    record_run(ctx.conn, s, "sync")
    return s


def retry_channel(cfg, slug, *, process=None) -> RunSummary:
    process = process or process_video
    info, filters = load_channel(cfg, slug)
    ctx = build_context(cfg, slug)
    todo = videos_by_state(ctx.conn, RETRY_STATES)
    from .models import VideoMeta
    results = []
    for row in todo:
        meta = VideoMeta(row["video_id"], row["title"], row["duration"], row["upload_date"], row["url"])
        results.append(process(ctx, meta))
    s = _summarize(results)
    record_run(ctx.conn, s, "retry")
    return s


def channel_status(cfg, slug) -> dict:
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    conn = connect(paths.db)
    return {"slug": slug, "counts": count_by_state(conn)}


def list_channels(cfg) -> list[str]:
    return list_channel_slugs(cfg.data_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sync.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/sync.py tests/test_sync.py
git commit -m "feat: channel add/sync/retry/status orchestration with run logging"
```

---

### Task 12: Retrieval tools for the agent

**Files:**
- Create: `ytkb/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `ytkb.store.ChannelStore`, `ytkb.models.ChunkHit`.
- Produces: `build_tools(store: ChannelStore, top_k: int) -> tuple[list[dict], Callable[[str, dict], str]]` returning `(tool_specs, dispatch)`. `tool_specs` is the OpenAI/OpenRouter tool-schema list for `keyword_search`, `semantic_search`, `read_transcript`, `list_videos`. `dispatch(name, args) -> str` runs the named tool and returns a JSON string the model can read. Each search result line includes `video_id`, `title`, `start`, and `text` so the agent can cite.

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval.py`:
```python
import json
from ytkb.retrieval import build_tools
from ytkb.models import ChunkHit


class FakeStore:
    def keyword_search(self, query, k):
        return [ChunkHit("v1", "T", 12.0, "cofounder advice", 1.0)]

    def semantic_search(self, query, k):
        return [ChunkHit("v2", "T2", 30.0, "co-founder concept", 0.1)]

    def read_around(self, video_id, around_ts, window=90.0):
        return "full context here"

    def list_videos(self, contains=None):
        return [{"video_id": "v1", "title": "T", "upload_date": "20240101"}]


def test_tool_specs_cover_all_tools():
    specs, _ = build_tools(FakeStore(), top_k=5)
    names = {s["function"]["name"] for s in specs}
    assert names == {"keyword_search", "semantic_search", "read_transcript", "list_videos"}


def test_dispatch_keyword_search_returns_json_with_citation_fields():
    _, dispatch = build_tools(FakeStore(), top_k=5)
    out = json.loads(dispatch("keyword_search", {"query": "cofounder"}))
    assert out[0]["video_id"] == "v1"
    assert out[0]["start"] == 12.0
    assert "text" in out[0]


def test_dispatch_read_transcript():
    _, dispatch = build_tools(FakeStore(), top_k=5)
    out = dispatch("read_transcript", {"video_id": "v1", "around_ts": 12.0})
    assert "full context" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/retrieval.py`**

```python
import json


def _hits_to_json(hits) -> str:
    return json.dumps([
        {"video_id": h.video_id, "title": h.title, "start": h.start, "text": h.text}
        for h in hits
    ])


def build_tools(store, top_k: int):
    def _spec(name, desc, props, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    specs = [
        _spec("keyword_search", "Exact keyword/BM25 search over transcript chunks. Best for names, jargon, exact phrases.",
              {"query": {"type": "string"}}, ["query"]),
        _spec("semantic_search", "Semantic search over transcript chunks. Best for concepts and paraphrases.",
              {"query": {"type": "string"}}, ["query"]),
        _spec("read_transcript", "Read transcript text for a video, optionally around a timestamp (seconds).",
              {"video_id": {"type": "string"}, "around_ts": {"type": "number"}}, ["video_id"]),
        _spec("list_videos", "List indexed videos; optionally filter by a substring of the title.",
              {"contains": {"type": "string"}}, []),
    ]

    def dispatch(name: str, args: dict) -> str:
        if name == "keyword_search":
            return _hits_to_json(store.keyword_search(args["query"], top_k))
        if name == "semantic_search":
            return _hits_to_json(store.semantic_search(args["query"], top_k))
        if name == "read_transcript":
            return store.read_around(args["video_id"], args.get("around_ts"), args.get("window", 90.0))
        if name == "list_videos":
            return json.dumps(store.list_videos(args.get("contains")))
        return json.dumps({"error": f"unknown tool {name}"})

    return specs, dispatch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/retrieval.py tests/test_retrieval.py
git commit -m "feat: agent retrieval tool specs and dispatch"
```

---

### Task 13: Agentic answer loop and citations

**Files:**
- Create: `ytkb/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `ytkb.llm.LLMClient`, `ytkb.retrieval.build_tools`, `ytkb.store.ChannelStore`, `ytkb.models` (`Answer`, `Citation`).
- Produces: `answer(question: str, channel_title: str, store: ChannelStore, llm: LLMClient, *, chat_model: str, top_k: int, max_steps: int = 5) -> Answer`. The loop sends the system prompt + question, executes tool calls the model requests (via `dispatch`), feeds results back, and stops when the model returns a final text answer or `max_steps` is hit. Citations are extracted from the tool results the model actually used (video_id+start → `https://youtu.be/<id>?t=<int(start)>`).

- [ ] **Step 1: Write the failing test (scripted fake LLM drives one tool call then answers)**

`tests/test_agent.py`:
```python
from types import SimpleNamespace
from ytkb.agent import answer
from ytkb.models import ChunkHit


class FakeStore:
    def keyword_search(self, query, k):
        return [ChunkHit("vid123", "Cofounders", 42.0, "find a technical cofounder", 1.0)]

    def semantic_search(self, query, k):
        return []

    def read_around(self, *a, **k):
        return ""

    def list_videos(self, contains=None):
        return []


class ScriptedLLM:
    """First call -> request keyword_search; second call -> final answer."""
    def __init__(self):
        self.n = 0

    def chat_with_tools(self, messages, model, tools):
        self.n += 1
        if self.n == 1:
            tc = SimpleNamespace(
                id="call1",
                function=SimpleNamespace(name="keyword_search", arguments='{"query": "cofounder"}'),
            )
            return SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(content="You should find a technical cofounder.", tool_calls=None)


def test_agent_runs_tool_then_answers_with_citation():
    ans = answer("How do I find a cofounder?", "Y Combinator", FakeStore(), ScriptedLLM(),
                 chat_model="m", top_k=5)
    assert "technical cofounder" in ans.text
    assert ans.citations[0].url == "https://youtu.be/vid123?t=42"


def test_agent_stops_without_tools():
    class DirectLLM:
        def chat_with_tools(self, messages, model, tools):
            return SimpleNamespace(content="Not covered in these videos.", tool_calls=None)

    ans = answer("unrelated?", "YC", FakeStore(), DirectLLM(), chat_model="m", top_k=5)
    assert ans.text.startswith("Not covered")
    assert ans.citations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/agent.py`**

```python
import json

from .models import Answer, Citation
from .retrieval import build_tools

SYSTEM = (
    "You are an expert assistant answering questions using ONLY the indexed videos from the "
    "YouTube channel '{channel}'. Use the search tools to find relevant transcript passages, "
    "read more context if needed, then answer concisely. Always ground claims in what the "
    "videos actually say. If the videos do not cover the question, say so plainly. "
    "Do as few searches as needed."
)


def answer(question, channel_title, store, llm, *, chat_model, top_k, max_steps=5) -> Answer:
    specs, dispatch = build_tools(store, top_k)
    messages = [
        {"role": "system", "content": SYSTEM.format(channel=channel_title)},
        {"role": "user", "content": question},
    ]
    cited: dict[str, Citation] = {}

    for _ in range(max_steps):
        msg = llm.chat_with_tools(messages, model=chat_model, tools=specs)
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return Answer(text=msg.content or "", citations=list(cited.values()))

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = dispatch(tc.function.name, args)
            _collect_citations(tc.function.name, result, cited)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # max_steps hit: ask for a final answer with no tools
    msg = llm.chat_with_tools(messages + [
        {"role": "user", "content": "Give your best final answer now using what you found."}
    ], model=chat_model, tools=[])
    return Answer(text=msg.content or "", citations=list(cited.values()))


def _collect_citations(tool_name: str, result: str, cited: dict) -> None:
    if tool_name not in ("keyword_search", "semantic_search"):
        return
    try:
        rows = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return
    for r in rows:
        vid, start = r.get("video_id"), r.get("start", 0.0)
        key = f"{vid}:{int(start)}"
        if vid and key not in cited:
            cited[key] = Citation(
                video_id=vid, title=r.get("title", ""), start=float(start),
                url=f"https://youtu.be/{vid}?t={int(start)}",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/agent.py tests/test_agent.py
git commit -m "feat: agentic tool-using answer loop with citations"
```

---

### Task 14: CLI

**Files:**
- Create: `ytkb/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ytkb.sync`, `ytkb.agent`, `ytkb.config.load_config`.
- Produces: a Typer `app` with commands `add`, `sync`, `ask`, `status`, `retry`, `list`, `remove`, `web`. `ask` builds a context via `sync.build_context`, calls `agent.answer`, and prints the answer + citations. Tests use Typer's `CliRunner` and monkeypatch `sync`/`agent` functions so no network/LLM is hit.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from typer.testing import CliRunner
from ytkb import cli
from ytkb.channel import ChannelInfo, ChannelFilters
from ytkb.models import Answer, Citation, RunSummary

runner = CliRunner()


def test_add_invokes_sync(monkeypatch, tmp_path):
    captured = {}

    def fake_add(cfg, url, filters, name=None, **k):
        captured["url"] = url
        captured["no_shorts"] = filters.no_shorts
        return "y-combinator"

    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli.sync, "add_channel", fake_add)
    result = runner.invoke(cli.app, ["add", "https://youtu.be/x", "--no-shorts"])
    assert result.exit_code == 0
    assert captured["url"] == "https://youtu.be/x"
    assert captured["no_shorts"] is True
    assert "y-combinator" in result.stdout


def test_ask_prints_answer_and_citations(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))

    class FakeCtx:
        store = object()
        llm = object()
        cfg = None

    info = ChannelInfo("UC1", "@yc", "Y Combinator", "u")
    monkeypatch.setattr(cli.sync, "build_context", lambda cfg, slug: FakeCtx())
    monkeypatch.setattr(cli.sync, "load_channel", lambda cfg, slug: (info, ChannelFilters()))
    monkeypatch.setattr(cli.agent, "answer", lambda *a, **k: Answer(
        "Find a technical cofounder.",
        [Citation("v1", "Cofounders", 42.0, "https://youtu.be/v1?t=42")],
    ))
    result = runner.invoke(cli.app, ["ask", "y-combinator", "How do I find a cofounder?"])
    assert result.exit_code == 0
    assert "Find a technical cofounder." in result.stdout
    assert "youtu.be/v1?t=42" in result.stdout


def test_status_prints_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli.sync, "channel_status", lambda cfg, slug: {"slug": slug, "counts": {"indexed": 3}})
    result = runner.invoke(cli.app, ["status", "y-combinator"])
    assert result.exit_code == 0
    assert "indexed" in result.stdout and "3" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/cli.py`**

```python
import typer

from . import sync, agent
from .channel import ChannelFilters
from .config import load_config

app = typer.Typer(help="YouTube channel knowledge base")


@app.command()
def add(url: str, max: int = typer.Option(None), since: str = typer.Option(None),
        min_length: int = typer.Option(None), no_shorts: bool = typer.Option(False),
        whisper: bool = typer.Option(True), name: str = typer.Option(None)):
    """Add a channel (does not sync)."""
    cfg = load_config()
    filters = ChannelFilters(max=max, since=since, min_length=min_length,
                             no_shorts=no_shorts, whisper=whisper)
    slug = sync.add_channel(cfg, url, filters, name=name)
    typer.echo(f"Added channel '{slug}'. Run: kb sync {slug}")


@app.command()
def sync_cmd(slug: str = typer.Argument(None), all: bool = typer.Option(False, "--all"),
             dry_run: bool = typer.Option(False, "--dry-run")):
    """Discover and process new/failed videos."""
    cfg = load_config()
    slugs = sync.list_channels(cfg) if all else [slug]
    for s in slugs:
        summary = sync.sync_channel(cfg, s, dry_run=dry_run)
        typer.echo(f"[{s}] new={summary.new} done={summary.done} "
                   f"failed={summary.failed} skipped={summary.skipped}")


# Typer maps function name 'sync_cmd' to command 'sync-cmd'; rename explicitly:
app.command(name="sync")(sync_cmd)


@app.command()
def ask(slug: str, question: str):
    """Ask a channel's agent a question."""
    cfg = load_config()
    info, _filters = sync.load_channel(cfg, slug)
    ctx = sync.build_context(cfg, slug)
    if ctx.llm is None:
        typer.echo("No OPENROUTER_API_KEY configured in data/.env", err=True)
        raise typer.Exit(1)
    ans = agent.answer(question, info.title, ctx.store, ctx.llm,
                       chat_model=cfg.chat_model, top_k=cfg.top_k)
    typer.echo(ans.text)
    if ans.citations:
        typer.echo("\nSources:")
        for c in ans.citations:
            typer.echo(f"  - {c.title} @ {int(c.start)}s  {c.url}")


@app.command()
def status(slug: str):
    cfg = load_config()
    st = sync.channel_status(cfg, slug)
    typer.echo(f"Channel: {st['slug']}")
    for state, n in sorted(st["counts"].items()):
        typer.echo(f"  {state}: {n}")


@app.command()
def retry(slug: str):
    cfg = load_config()
    summary = sync.retry_channel(cfg, slug)
    typer.echo(f"[{slug}] done={summary.done} failed={summary.failed} skipped={summary.skipped}")


@app.command(name="list")
def list_cmd():
    cfg = load_config()
    for s in sync.list_channels(cfg):
        typer.echo(s)


@app.command()
def remove(slug: str, yes: bool = typer.Option(False, "--yes")):
    import shutil
    from .paths import ChannelPaths
    cfg = load_config()
    if not yes:
        typer.confirm(f"Delete all data for '{slug}'?", abort=True)
    shutil.rmtree(ChannelPaths.for_slug(cfg.data_dir, slug).dir, ignore_errors=True)
    typer.echo(f"Removed {slug}")


@app.command()
def web(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    from .web.app import create_app
    uvicorn.run(create_app(), host=host, port=port)
```

> Note: the `sync` command is registered twice in the snippet (the `@app.command()` decorator plus the explicit `app.command(name="sync")(sync_cmd)`). During implementation, drop the bare `@app.command()` decorator on `sync_cmd` and keep only the explicit `app.command(name="sync")(sync_cmd)` registration so there is exactly one `sync` command.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/cli.py tests/test_cli.py
git commit -m "feat: Typer CLI (add/sync/ask/status/retry/list/remove/web)"
```

---

### Task 15: Web chat UI

**Files:**
- Create: `ytkb/web/__init__.py`, `ytkb/web/app.py`, `ytkb/web/static/index.html`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `ytkb.sync`, `ytkb.agent`, `ytkb.config.load_config`.
- Produces: `create_app(cfg=None) -> FastAPI`. Routes: `GET /` serves the static page; `GET /api/channels` returns `{"channels": [...]}`; `POST /api/ask` with `{"slug": ..., "question": ...}` returns `{"answer": ..., "citations": [...]}`. (Non-streaming JSON keeps it testable; the page may poll. Streaming is an optional later enhancement.)

- [ ] **Step 1: Write the failing test (FastAPI TestClient, agent monkeypatched)**

`tests/test_web.py`:
```python
from fastapi.testclient import TestClient
from ytkb.web.app import create_app
from ytkb.channel import ChannelInfo, ChannelFilters
from ytkb.models import Answer, Citation
from ytkb import sync, agent
import ytkb.web.app as webapp


def test_channels_and_ask(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(webapp.sync, "list_channels", lambda cfg: ["y-combinator"])

    class FakeCtx:
        store = object()
        llm = object()

    info = ChannelInfo("UC1", "@yc", "Y Combinator", "u")
    monkeypatch.setattr(webapp.sync, "build_context", lambda cfg, slug: FakeCtx())
    monkeypatch.setattr(webapp.sync, "load_channel", lambda cfg, slug: (info, ChannelFilters()))
    monkeypatch.setattr(webapp.agent, "answer", lambda *a, **k: Answer(
        "Find a cofounder.", [Citation("v1", "Cofounders", 42.0, "https://youtu.be/v1?t=42")]))

    client = TestClient(create_app())
    assert client.get("/api/channels").json() == {"channels": ["y-combinator"]}

    r = client.post("/api/ask", json={"slug": "y-combinator", "question": "how?"})
    body = r.json()
    assert body["answer"] == "Find a cofounder."
    assert body["citations"][0]["url"] == "https://youtu.be/v1?t=42"


def test_index_served(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ytkb/web/app.py` and the static page**

`ytkb/web/__init__.py`: empty file.

`ytkb/web/app.py`:
```python
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import __init__  # noqa
from .. import sync, agent
from ..config import load_config

STATIC = Path(__file__).parent / "static"


class AskReq(BaseModel):
    slug: str
    question: str


def create_app(cfg=None) -> FastAPI:
    app = FastAPI(title="ytkb")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/channels")
    def channels():
        return {"channels": sync.list_channels(cfg or load_config())}

    @app.post("/api/ask")
    def ask(req: AskReq):
        c = cfg or load_config()
        info, _ = sync.load_channel(c, req.slug)
        ctx = sync.build_context(c, req.slug)
        ans = agent.answer(req.question, info.title, ctx.store, ctx.llm,
                           chat_model=c.chat_model, top_k=c.top_k)
        return {"answer": ans.text, "citations": [asdict(x) for x in ans.citations]}

    return app
```

> Note: remove the `from . import __init__` line during implementation — it is a no-op import artifact. `create_app` reads `cfg` once per request via `load_config()` when not provided.

`ytkb/web/static/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>YouTube Channel KB</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
    #log { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; min-height: 240px; }
    .q { font-weight: 600; margin-top: 1rem; }
    .a { white-space: pre-wrap; }
    .src a { display: block; font-size: 0.85rem; color: #2a6; }
    textarea { width: 100%; height: 64px; }
    select, button { padding: 0.4rem 0.6rem; }
  </style>
</head>
<body>
  <h1>Channel Knowledge Base</h1>
  <select id="channel"></select>
  <div id="log"></div>
  <textarea id="q" placeholder="Ask a question..."></textarea>
  <button id="send">Ask</button>
  <script>
    async function loadChannels() {
      const r = await fetch('/api/channels'); const d = await r.json();
      const sel = document.getElementById('channel');
      sel.innerHTML = d.channels.map(c => `<option>${c}</option>`).join('');
    }
    async function ask() {
      const slug = document.getElementById('channel').value;
      const question = document.getElementById('q').value.trim();
      if (!question) return;
      const log = document.getElementById('log');
      log.innerHTML += `<div class="q">Q: ${question}</div><div class="a">…</div>`;
      const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({slug, question})});
      const d = await r.json();
      const srcs = (d.citations||[]).map(c => `<a href="${c.url}" target="_blank">${c.title} @ ${Math.floor(c.start)}s</a>`).join('');
      log.lastChild.innerHTML = d.answer + `<div class="src">${srcs}</div>`;
      document.getElementById('q').value = '';
    }
    document.getElementById('send').onclick = ask;
    loadChannels();
  </script>
</body>
</html>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -v`
Expected: PASS (all tests green)

```bash
git add ytkb/web/ tests/test_web.py
git commit -m "feat: FastAPI web chat UI with channel picker and citations"
```

---

### Task 16: End-to-end smoke test and README

**Files:**
- Create: `README.md`
- Modify: `data/config.toml.example` (create example config)
- Test: manual (documented below)

**Interfaces:** none (documentation + manual verification).

- [ ] **Step 1: Write a default example config**

Create `data/config.toml.example`:
```toml
# Copy to data/config.toml and edit as needed.
chat_model = "anthropic/claude-haiku-4.5"
adstrip_model = "anthropic/claude-haiku-4.5"
embedding_model = "BAAI/bge-small-en-v1.5"
chunk_tokens = 500
chunk_overlap = 80
languages = ["en"]
whisper_enabled_default = true
whisper_model = "base"
request_delay = 1.0
top_k = 6
```

- [ ] **Step 2: Write `README.md`** with setup + usage:

````markdown
# YouTube Channel Knowledge Base (`ytkb`)

Local tool: ingest a YouTube channel's transcripts, strip sponsor reads, and ask a
per-channel agent questions with citations.

## Setup
```bash
uv sync
echo "OPENROUTER_API_KEY=sk-..." > data/.env
cp data/config.toml.example data/config.toml   # optional
```

## Usage
```bash
uv run kb add "https://www.youtube.com/@ycombinator" --no-shorts --name ycombinator
uv run kb sync ycombinator            # add --dry-run to preview
uv run kb status ycombinator
uv run kb ask ycombinator "How do I find a co-founder?"
uv run kb retry ycombinator           # re-process failed/no-transcript
uv run kb web                         # open http://127.0.0.1:8000
```

Data lives under `data/channels/<slug>/`. Back up by copying `data/`.
````

- [ ] **Step 3: Manual smoke test against a small real channel**

Run (requires real `OPENROUTER_API_KEY` and network):
```bash
uv run kb add "https://www.youtube.com/@ycombinator" --no-shorts --max 3 --name yc-smoke
uv run kb sync yc-smoke
uv run kb status yc-smoke       # expect ~3 indexed
uv run kb ask yc-smoke "What advice is given about co-founders?"
```
Expected: status shows indexed videos; `ask` prints an answer plus at least one
`youtu.be/<id>?t=<sec>` citation. Verify a citation link opens at the right moment.

- [ ] **Step 4: Commit**

```bash
git add README.md data/config.toml.example
git commit -m "docs: README, example config, and end-to-end smoke test"
```

> Note: `data/` is gitignored; force-add only the example file:
> `git add -f data/config.toml.example`.

---

## Self-Review

**Spec coverage check (spec section → task):**
- §4 data layout → Tasks 1, 2, 9, 11 (paths, db, store, channel.json).
- §5 ingestion pipeline (resolve, list, fetch, whisper, ad-strip, chunk, index) → Tasks 3–10.
- §6 state machine & logging → Tasks 2, 10, 11 (`runs` table, `record_run`).
- §7 agentic RAG + 4 tools + citations → Tasks 12, 13.
- §8 CLI commands (add/sync/ask/status/retry/list/web/remove) → Task 14.
- §9 web UI → Task 15.
- §10 config (config.toml + .env, swappable model) → Task 1, 16.
- §11 cross-cutting: channel resolution (Task 3), rate-limit `request_delay` config (Tasks 1; applied in yt-dlp opts during Task 3/4 implementation), `--dry-run` (Task 11/14), LLM-only-when-SponsorBlock-empty (Task 7), dedup by video_id (Task 2 PK + Task 11 new-detection), language config (Tasks 1, 4), portability (Task 1 layout), ToS note (README, Task 16).
- §12 testing strategy → every task is TDD; manual smoke in Task 16.

**Gaps fixed inline:** `request_delay` is defined in config (Task 1) and should be threaded into the yt-dlp option dicts (`sleep_interval`) in `channel.py`/`transcripts.py` when implementing Tasks 3–4 — noted here so it is not missed. Re-sync deleted/private-video handling: `list_videos` simply won't return them, so they keep their last state; no extra task needed (acceptable per spec).

**Placeholder scan:** no TBD/TODO; the three "Note" callouts intentionally flag dead-code lines to delete during implementation (LanceDB `read_around`, double-registered `sync` command, web `__init__` import). These are explicit instructions, not placeholders.

**Type consistency:** `VideoState` enum values, `ChannelFilters` fields, `ChunkHit`/`Citation`/`Answer` shapes, and `ChannelStore`/`build_tools`/`answer` signatures are consistent across Tasks 2, 9, 12, 13, 14, 15.
