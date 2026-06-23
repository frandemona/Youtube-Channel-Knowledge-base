# `kb reindex` Command — Design Spec

**Date:** 2026-06-23
**Status:** Approved (pending final spec review)
**Builds on:** [2026-06-20-youtube-channel-knowledgebase-design.md](2026-06-20-youtube-channel-knowledgebase-design.md)

## 1. Purpose

Let a user rebuild a channel's local index **without re-downloading transcripts or re-running
ad-stripping** — primarily so they can switch the local `embedding_model` (or change
`chunk_tokens`/`chunk_overlap`) in `config.toml` cheaply. Today the only path is
`kb remove` + `kb add` + `kb sync`, which re-hits YouTube and OpenRouter unnecessarily.

Re-embedding is a purely **local, offline** operation: the cleaned transcript already lives on
disk after the first sync.

Non-goals (YAGNI): re-fetching captions, re-running ad detection, per-channel embedding configs,
incremental/partial reindex, a progress bar.

## 2. The timestamp problem and the chosen fix (A2)

Citations need a per-chunk `start` time to build `https://youtu.be/<id>?t=<sec>` links. The
current cleaned artifact `<id>.clean.txt` is **flat text with no timestamps**, so re-chunking
from it would lose all timestamps (every chunk `start=0.0`).

**Fix (A2):** persist the cleaned transcript as **timestamped segments** in a new
`<id>.clean.json` (a `list[Segment]`, same JSON shape `save_raw`/`load_raw` already use),
written during ingestion. `reindex` re-chunks from `clean.json`, preserving timestamps exactly,
fully offline. `clean.txt` (human-readable joined text) continues to be written as before.

## 3. Changes

### 3.1 Paths — `ytkb/paths.py`
Add `ChannelPaths.clean_segments_path(video_id) -> Path` returning
`transcripts/<video_id>.clean.json`.

### 3.2 Pipeline — `ytkb/pipeline.py`
In `process_video`, after ad-stripping produces `clean_segments`, in addition to the existing
`clean.txt` write, persist the timestamped cleaned segments:
`transcripts.save_raw(ctx.paths.clean_segments_path(vid), clean_segments)`.
(Order relative to `clean.txt` does not matter; both happen before `set_state(AD_STRIPPED)`.)

### 3.3 Store reset — `ytkb/store.py`
Add `ChannelStore.reset() -> None` that fully clears the channel's index in BOTH stores so a
new embedding model (likely a different vector dimension) can be written cleanly:
- **LanceDB:** drop the `chunks` table if it exists (`self._db.drop_table(TABLE)`). A full drop
  is required — appending vectors of a new dimension to an existing table fails.
- **SQLite/FTS5:** clear the FTS index then the content table:
  `INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');` then `DELETE FROM chunks;` (commit).
`reset()` is idempotent (safe when the table/rows don't exist).

### 3.4 Reindex orchestration — `ytkb/sync.py`
Add `reindex_channel(cfg, slug, *, process=None) -> RunSummary`:
1. `ctx = build_context(cfg, slug)` (a fresh `Embedder` reads the current `embedding_model`).
2. `ctx.store.reset()`.
3. For each video in state `INDEXED` (`videos_by_state(ctx.conn, [VideoState.INDEXED])`):
   - Reindex it via an injectable `process(ctx, row)` callable (default: a module-level
     `reindex_video`), which:
     - loads cleaned segments: prefer `clean_segments_path(vid)` (timestamped); if absent, fall
       back to wrapping `clean_path(vid)` text as a single `Segment(0.0, 0.0, text)` (legacy
       channels indexed before `clean.json` existed — content still searchable, timestamps lost);
       if neither file exists, skip (count as skipped).
     - `chunk_segments(vid, segments, cfg.chunk_tokens, cfg.chunk_overlap)` → `store.add(chunks, {vid: row["title"]})`.
     - on success leave/keep state `INDEXED`; on exception set `FAILED_EMBED` and count failed.
4. `record_run(ctx.conn, summary, "reindex")`; return the `RunSummary` (done/failed/skipped).

### 3.5 CLI — `ytkb/cli.py`
Add `reindex` command: `kb reindex [<slug>] [--all]`, mirroring `sync`'s slug/`--all` handling,
printing `[<slug>] reindexed=<done> failed=<failed> skipped=<skipped>`.

## 4. Error handling & edge cases
- Channel/video with no cleaned transcript on disk → skipped, counted, never crashes.
- `reset()` on a channel that was never indexed (no table/rows) → no-op.
- An embed failure on one video sets `FAILED_EMBED` for that video and continues; the user can
  `kb retry` (which re-runs the full pipeline) or `kb reindex` again.
- Because `reset()` clears the whole index first, a reindex is atomic-by-rebuild: a crash
  mid-reindex leaves a partial index, but re-running `reindex` starts from a clean reset.

## 5. Testing
- `store.reset()`: index chunks, `reset()`, assert `keyword_search` returns `[]`, `semantic_search`
  returns `[]`, and the LanceDB table is gone; calling `reset()` again does not error.
- `pipeline`: after `process_video`, assert `<id>.clean.json` exists and round-trips to the
  cleaned segments (with timestamps), alongside the existing `clean.txt`.
- `reindex_channel`: index a channel with a fake embedder of dim N; change to a fake embedder of
  dim M; `reindex_channel`; assert vectors are now dim M, chunk counts are stable (not doubled),
  keyword search still works, and the run is recorded with kind `reindex`.
- `reindex_channel` skip path: a video in `INDEXED` state with no `clean.json`/`clean.txt` is
  counted as skipped and does not crash.
- CLI: `kb reindex <slug>` invokes `sync.reindex_channel` and prints the summary
  (monkeypatched, no real work).

## 6. Out of scope / follow-ups (unchanged from parent spec)
`last_sync`, run-error persistence, SSE streaming, multilingual embedding defaults. A `kb reindex`
does not address these.
