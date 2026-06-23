# `kb reindex` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `kb reindex <slug>` to rebuild a channel's local search index (chunks + FTS5 + LanceDB vectors) from on-disk cleaned transcripts — offline, no re-download — so the embedding model or chunk settings can be changed cheaply.

**Architecture:** Ingestion gains a new on-disk artifact `<id>.clean.json` (cleaned, still-timestamped segments) so reindex preserves citation timestamps. `ChannelStore.reset()` fully drops the channel's index; `sync.reindex_channel` resets then re-chunks + re-embeds every indexed video from `clean.json`; a Typer `reindex` command wires it up.

**Tech Stack:** Existing `ytkb` stack — Python 3.12, uv, SQLite+FTS5, LanceDB, fastembed, Typer. No new dependencies.

## Global Constraints

- Python **3.12**; manage with **uv** (`uv run pytest`, `uv run kb`).
- Package `ytkb`; flat layout; tests under `tests/` (pytest `pythonpath=["."]`, `tests/__init__.py` exists).
- TDD every task: write failing test → run it, see it fail → implement → run, see it pass → commit.
- No network/LLM/real-embedding-model loads in tests. Use the existing `HashBackend` fake embedder (`tests/test_store.py`) and injected callables. `fastembed` is lazily imported inside `Embedder`, so merely *constructing* `Embedder(...)` (as `build_context` does) loads no model — only an actual `.embed()` call would.
- Cleaned segments serialize with the existing `transcripts.save_raw(path, list[Segment])` / `transcripts.load_raw(path) -> list[Segment]` (JSON of `{start,end,text}`).
- Reuse existing helpers; do not duplicate logic (DRY). Build only what the spec lists (YAGNI).

---

### Task 1: Persist timestamped cleaned segments (`clean.json`)

**Files:**
- Modify: `ytkb/paths.py` (add `clean_segments_path`, after `clean_path` at line 50-51)
- Modify: `ytkb/pipeline.py` (after the `clean.txt` write, lines 57-59)
- Test: `tests/test_paths.py` (add one test), `tests/test_pipeline.py` (add one test)

**Interfaces:**
- Consumes: `ytkb.transcripts.save_raw(path, segments)` / `load_raw(path)`; `ChannelPaths`.
- Produces: `ChannelPaths.clean_segments_path(video_id: str) -> Path` → `transcripts/<video_id>.clean.json`. Ingestion now writes that file (the cleaned, timestamped `list[Segment]`) in addition to the existing `clean.txt`.

- [ ] **Step 1: Write the failing path test**

Add to `tests/test_paths.py` (inside `test_channel_paths_layout`, after the existing `clean_path` assertion):
```python
    assert cp.clean_segments_path("abc123") == cp.transcripts_dir / "abc123.clean.json"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_paths.py::test_channel_paths_layout -v`
Expected: FAIL with `AttributeError: 'ChannelPaths' object has no attribute 'clean_segments_path'`

- [ ] **Step 3: Implement `clean_segments_path`**

In `ytkb/paths.py`, add this method immediately after `clean_path` (after line 51):
```python
    def clean_segments_path(self, video_id: str) -> Path:
        return self.transcripts_dir / f"{video_id}.clean.json"
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing pipeline test**

Add to `tests/test_pipeline.py`:
```python
def test_pipeline_writes_timestamped_clean_json(tmp_path):
    from ytkb.transcripts import load_raw
    ctx = make_ctx(tmp_path)
    upsert_video(ctx.conn, meta())
    segs = [Segment(10.0, 12.0, "find a cofounder"), Segment(12.0, 15.0, "this is sponsored")]
    process_video(
        ctx, meta(),
        fetch=lambda vid, langs: segs,
        strip=lambda vid, s, llm, cfg, **k: ([segs[0]], "llm"),
    )
    clean_json = ctx.paths.clean_segments_path("v1")
    assert clean_json.exists()
    loaded = load_raw(clean_json)
    assert loaded == [Segment(10.0, 12.0, "find a cofounder")]  # cleaned + timestamp preserved
```

- [ ] **Step 6: Run it, verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_pipeline_writes_timestamped_clean_json -v`
Expected: FAIL — `clean_json.exists()` is False (file not written yet).

- [ ] **Step 7: Implement the pipeline write**

In `ytkb/pipeline.py`, the current block (lines 57-60) is:
```python
    ctx.paths.clean_path(vid).write_text(
        " ".join(s.text for s in clean_segments), encoding="utf-8"
    )
    set_state(ctx.conn, vid, VideoState.AD_STRIPPED)
```
Insert a `save_raw` call for the timestamped segments between the `clean.txt` write and `set_state`:
```python
    ctx.paths.clean_path(vid).write_text(
        " ".join(s.text for s in clean_segments), encoding="utf-8"
    )
    transcripts.save_raw(ctx.paths.clean_segments_path(vid), clean_segments)
    set_state(ctx.conn, vid, VideoState.AD_STRIPPED)
```
(`transcripts` is already imported at the top of `pipeline.py`.)

- [ ] **Step 8: Run tests, verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_paths.py -v`
Expected: PASS (existing pipeline/path tests still green too).

- [ ] **Step 9: Commit**

```bash
git add ytkb/paths.py ytkb/pipeline.py tests/test_paths.py tests/test_pipeline.py
git commit -m "feat: persist timestamped cleaned segments as clean.json"
```

---

### Task 2: `ChannelStore.reset()` + `db.clear_all_chunks`

**Files:**
- Modify: `ytkb/db.py` (add `clear_all_chunks`)
- Modify: `ytkb/store.py` (add `reset` method)
- Test: `tests/test_db.py` (one test), `tests/test_store.py` (one test)

**Interfaces:**
- Consumes: `lancedb` table API; the `chunks`/`chunks_fts` schema.
- Produces:
  - `ytkb.db.clear_all_chunks(conn) -> None` — empties `chunks` and the FTS5 index (`chunks_fts`).
  - `ytkb.store.ChannelStore.reset() -> None` — drops the LanceDB `chunks` table (if present) and calls `clear_all_chunks`. Idempotent.

- [ ] **Step 1: Write the failing db test**

Add to `tests/test_db.py`:
```python
def test_clear_all_chunks_empties_content_and_fts(tmp_path):
    from ytkb.db import clear_all_chunks
    conn = connect(tmp_path / "videos.db")
    upsert_video(conn, make_meta("v1"))
    insert_chunks(conn, [Chunk(video_id="v1", idx=0, start=1.0, text="cofounder advice")], {"v1": "T"})
    clear_all_chunks(conn)
    assert conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 0
    # FTS index is empty too -> a prior match now returns nothing
    rows = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("cofounder",)).fetchall()
    assert rows == []
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_db.py::test_clear_all_chunks_empties_content_and_fts -v`
Expected: FAIL with `ImportError: cannot import name 'clear_all_chunks'`

- [ ] **Step 3: Implement `clear_all_chunks`**

In `ytkb/db.py`, add (near `delete_video_chunks`):
```python
def clear_all_chunks(conn) -> None:
    # FTS5 external-content: clear the index first, then the content table.
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
    conn.execute("DELETE FROM chunks")
    conn.commit()
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing store test**

Add to `tests/test_store.py` (reuses `build_store` + `HashBackend` already in this file):
```python
def test_reset_clears_index(tmp_path):
    store = build_store(tmp_path)
    assert store.keyword_search("cofounder", 5) != []
    assert store.semantic_search("cofounder", 5) != []
    store.reset()
    assert store.keyword_search("cofounder", 5) == []
    assert store.semantic_search("cofounder", 5) == []
    assert store._table() is None
    store.reset()  # idempotent: no error when already empty / table absent
```

- [ ] **Step 6: Run it, verify it fails**

Run: `uv run pytest tests/test_store.py::test_reset_clears_index -v`
Expected: FAIL with `AttributeError: 'ChannelStore' object has no attribute 'reset'`

- [ ] **Step 7: Implement `reset`**

In `ytkb/store.py`, add `clear_all_chunks` to the existing import from `.db` (line 3 becomes):
```python
from .db import clear_all_chunks, delete_video_chunks, insert_chunks
```
Then add this method to `ChannelStore` (e.g. after `add`):
```python
    def reset(self) -> None:
        """Drop the entire index for this channel (vectors + chunks + FTS). Idempotent."""
        if TABLE in self._db.table_names():
            self._db.drop_table(TABLE)
        clear_all_chunks(self.conn)
```

- [ ] **Step 8: Run tests, verify they pass**

Run: `uv run pytest tests/test_store.py tests/test_db.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add ytkb/db.py ytkb/store.py tests/test_db.py tests/test_store.py
git commit -m "feat: ChannelStore.reset() to fully drop a channel index"
```

---

### Task 3: `reindex_channel` + `reindex_video`

**Files:**
- Modify: `ytkb/sync.py` (add `reindex_video` and `reindex_channel`; add `Segment` import)
- Test: `tests/test_sync.py` (three tests)

**Interfaces:**
- Consumes: `build_context`, `ChannelStore.reset`, `db.videos_by_state`, `db.set_state`, `db.record_run`, `VideoState`, `chunking.chunk_segments`, `transcripts.load_raw`, `models.Segment`, `models.RunSummary`, `ChannelPaths.clean_segments_path`/`clean_path`.
- Produces:
  - `ytkb.sync.reindex_video(ctx: ChannelContext, row) -> VideoState | None` — re-chunks + re-embeds one already-cleaned video from disk. Returns `VideoState.INDEXED` on success, `VideoState.FAILED_EMBED` on error (and sets that state), or `None` when the video has no cleaned transcript on disk (caller counts it skipped). `row` is a `sqlite3.Row` from the `videos` table.
  - `ytkb.sync.reindex_channel(cfg, slug, *, process=None) -> RunSummary` — `process` defaults to `reindex_video`; resets the index, reprocesses every `INDEXED` video, records a `"reindex"` run.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sync.py` (top of file already has `from ytkb.config import load_config`, `from ytkb.channel import ChannelInfo, ChannelFilters`, `from ytkb.db import VideoState, get_video`, `from ytkb.models import VideoMeta`, `from ytkb import sync`, and an `info()` helper):
```python
def _indexed_ctx_with_clean_json(tmp_path):
    """Build a channel whose video 'v1' is INDEXED and has a timestamped clean.json on disk."""
    from ytkb.paths import ChannelPaths
    from ytkb.db import connect, upsert_video, set_state
    from ytkb.embeddings import Embedder
    from ytkb.store import ChannelStore
    from ytkb.pipeline import ChannelContext
    from ytkb.transcripts import save_raw
    from ytkb.models import Segment
    from tests.test_store import HashBackend

    cfg = load_config(tmp_path)
    paths = ChannelPaths.for_slug(cfg.data_dir, "c")
    paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v1", "Cofounders", 600, "20240101", "https://youtu.be/v1"))
    set_state(conn, "v1", VideoState.INDEXED)
    save_raw(paths.clean_segments_path("v1"), [Segment(10.0, 12.0, "find a cofounder")])
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    ctx = ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=None, store=store,
                         filters=ChannelFilters(), whisper_enabled=True)
    return ctx


def test_reindex_video_rebuilds_from_clean_json(tmp_path):
    ctx = _indexed_ctx_with_clean_json(tmp_path)
    row = get_video(ctx.conn, "v1")
    state = sync.reindex_video(ctx, row)
    assert state == VideoState.INDEXED
    hits = ctx.store.keyword_search("cofounder", 5)
    assert hits[0].video_id == "v1"
    assert hits[0].start == 10.0  # timestamp preserved from clean.json


def test_reindex_video_skips_when_no_clean_files(tmp_path):
    from ytkb.paths import ChannelPaths
    from ytkb.db import connect, upsert_video, set_state
    from ytkb.embeddings import Embedder
    from ytkb.store import ChannelStore
    from ytkb.pipeline import ChannelContext
    from tests.test_store import HashBackend
    cfg = load_config(tmp_path)
    paths = ChannelPaths.for_slug(cfg.data_dir, "c"); paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v9", "No files", 600, "20240101", "https://youtu.be/v9"))
    set_state(conn, "v9", VideoState.INDEXED)
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    ctx = ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=None, store=store,
                         filters=ChannelFilters(), whisper_enabled=True)
    assert sync.reindex_video(ctx, get_video(conn, "v9")) is None


def test_reindex_channel_resets_processes_indexed_only_and_logs(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    ctx = sync.build_context(cfg, slug)
    from ytkb.db import upsert_video, set_state, connect
    from ytkb.paths import ChannelPaths
    upsert_video(ctx.conn, VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a"))
    set_state(ctx.conn, "a", VideoState.INDEXED)
    upsert_video(ctx.conn, VideoMeta("b", "B", 600, "20240102", "https://youtu.be/b"))
    set_state(ctx.conn, "b", VideoState.FAILED_FETCH)  # must NOT be reindexed

    processed = []
    def fake_process(ctx, row):
        processed.append(row["video_id"]); return VideoState.INDEXED

    summary = sync.reindex_channel(cfg, slug, process=fake_process)
    assert processed == ["a"]
    assert summary.done == 1
    # a "reindex" run was logged
    conn2 = connect(ChannelPaths.for_slug(cfg.data_dir, slug).db)
    kinds = [r["kind"] for r in conn2.execute("SELECT kind FROM runs").fetchall()]
    assert "reindex" in kinds
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_sync.py -k reindex -v`
Expected: FAIL with `AttributeError: module 'ytkb.sync' has no attribute 'reindex_video'`

- [ ] **Step 3: Implement `reindex_video` and `reindex_channel`**

In `ytkb/sync.py`: extend the chunking/transcripts imports and add the functions. First add these imports near the top (after the existing imports):
```python
from .chunking import chunk_segments
from .transcripts import load_raw
from .models import Segment
```
(`RunSummary`, `VideoMeta`, `VideoState`, `videos_by_state`, `record_run`, `build_context` are already imported. Add `set_state` to the existing `from .db import ...` line.)

Then add:
```python
def reindex_video(ctx: ChannelContext, row) -> VideoState | None:
    vid = row["video_id"]
    seg_path = ctx.paths.clean_segments_path(vid)
    if seg_path.exists():
        segments = load_raw(seg_path)
    elif ctx.paths.clean_path(vid).exists():
        # Legacy channels indexed before clean.json: text only, timestamps unavailable.
        segments = [Segment(0.0, 0.0, ctx.paths.clean_path(vid).read_text())]
    else:
        return None
    try:
        chunks = chunk_segments(vid, segments, ctx.cfg.chunk_tokens, ctx.cfg.chunk_overlap)
        ctx.store.add(chunks, title_of={vid: row["title"]})
    except Exception as e:
        set_state(ctx.conn, vid, VideoState.FAILED_EMBED, error=str(e))
        return VideoState.FAILED_EMBED
    return VideoState.INDEXED


def reindex_channel(cfg, slug, *, process=None) -> RunSummary:
    process = process or reindex_video
    ctx = build_context(cfg, slug)
    ctx.store.reset()
    summary = RunSummary()
    for row in videos_by_state(ctx.conn, [VideoState.INDEXED]):
        res = process(ctx, row)
        if res == VideoState.INDEXED:
            summary.done += 1
        elif res is None:
            summary.skipped += 1
        else:
            summary.failed += 1
    record_run(ctx.conn, summary, "reindex")
    return summary
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_sync.py -v`
Expected: PASS (all sync tests, new and existing).

- [ ] **Step 5: Commit**

```bash
git add ytkb/sync.py tests/test_sync.py
git commit -m "feat: reindex_channel/reindex_video — offline re-embed from clean.json"
```

---

### Task 4: `kb reindex` CLI command

**Files:**
- Modify: `ytkb/cli.py` (add the `reindex` command after the `retry` command)
- Test: `tests/test_cli.py` (one test)

**Interfaces:**
- Consumes: `sync.reindex_channel`, `sync.list_channels`, `RunSummary` (`.done/.failed/.skipped`).
- Produces: CLI command `reindex` — `kb reindex [<slug>] [--all]`, printing `[<slug>] reindexed=<done> failed=<failed> skipped=<skipped>` per channel.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py` (file already has `from typer.testing import CliRunner`, `from ytkb import cli`, `from ytkb.models import ... RunSummary`, `runner = CliRunner()`):
```python
def test_reindex_invokes_sync(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    captured = {}
    def fake_reindex(cfg, slug):
        captured["slug"] = slug
        return RunSummary(done=3, failed=0, skipped=1)
    monkeypatch.setattr(cli.sync, "reindex_channel", fake_reindex)
    result = runner.invoke(cli.app, ["reindex", "y-combinator"])
    assert result.exit_code == 0
    assert captured["slug"] == "y-combinator"
    assert "reindexed=3" in result.stdout
    assert "skipped=1" in result.stdout
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_cli.py::test_reindex_invokes_sync -v`
Expected: FAIL — `reindex` is not a known command (exit code 2 / "No such command").

- [ ] **Step 3: Implement the `reindex` command**

In `ytkb/cli.py`, add after the `retry` command (after line 68):
```python
@app.command()
def reindex(slug: str = typer.Argument(None), all: bool = typer.Option(False, "--all")):
    """Rebuild a channel's index from local transcripts (e.g. after changing the embedding model)."""
    cfg = load_config()
    slugs = sync.list_channels(cfg) if all else [slug]
    for s in slugs:
        summary = sync.reindex_channel(cfg, s)
        typer.echo(f"[{s}] reindexed={summary.done} failed={summary.failed} skipped={summary.skipped}")
```

- [ ] **Step 4: Run it, verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS. Also sanity-check the command is registered: `uv run kb --help` lists `reindex`.

- [ ] **Step 5: Commit**

```bash
git add ytkb/cli.py tests/test_cli.py
git commit -m "feat: kb reindex command"
```

---

### Task 5: Docs — README + spec cross-reference

**Files:**
- Modify: `README.md` (add `reindex` to the usage block + a one-line "changing the embedding model" note)
- Test: none (docs)

**Interfaces:** none.

- [ ] **Step 1: Update the README usage block**

In `README.md`, in the `## Usage` fenced block, add a line near `kb retry`:
```bash
uv run kb reindex ycombinator      # rebuild the index from local transcripts (e.g. after changing embedding_model)
```

- [ ] **Step 2: Add a short note under Usage**

Add this paragraph after the usage block:
```markdown
### Changing the embedding model
Edit `embedding_model` in `data/config.toml`, then run `uv run kb reindex <slug>`. This rebuilds
the vector index from the cleaned transcripts already on disk — no re-download, no LLM calls.
Switching embedding models requires a reindex because the vector dimension changes.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document kb reindex and embedding-model changes"
```

---

## Self-Review

**Spec coverage (spec section → task):**
- §2 / §3.1 / §3.2 (timestamped `clean.json` + `clean_segments_path` + pipeline write) → Task 1.
- §3.3 (`store.reset()` drops LanceDB table + clears FTS/chunks) → Task 2.
- §3.4 (`reindex_channel`/`reindex_video`, INDEXED-only, clean.json-with-clean.txt-fallback, `record_run("reindex")`) → Task 3.
- §3.5 (CLI `reindex` with `--all`) → Task 4.
- §4 error/edge cases (missing files → skip; reset idempotent; failed embed → FAILED_EMBED) → covered by Task 2 idempotent test + Task 3 skip/fail handling.
- §5 testing → each task is TDD; the dim-preserving timestamp assertion (`start == 10.0`) in Task 3 proves the A2 fix end to end.
- §6 (out of scope) → nothing built for those, correct.

**Placeholder scan:** no TBD/TODO; every code step shows complete code.

**Type consistency:** `clean_segments_path(video_id) -> Path`, `clear_all_chunks(conn)`, `ChannelStore.reset()`, `reindex_video(ctx, row) -> VideoState | None`, `reindex_channel(cfg, slug, *, process=None) -> RunSummary` are used identically across Tasks 1-4 and the tests. `RunSummary` fields `done/failed/skipped` match the existing dataclass. `store.add(chunks, title_of=...)`, `chunk_segments(vid, segments, chunk_tokens, chunk_overlap)`, `load_raw/save_raw` signatures match the existing code read during planning.
