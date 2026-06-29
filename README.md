# ytkb · YouTube Channel Knowledge Base

> Turn any YouTube channel into a local, searchable knowledge base you can chat with.

Point `ytkb` at a channel and it downloads every video's transcript, strips the sponsor reads,
and indexes it on your machine. Then you ask a per-channel agent questions and get answers
grounded in what the videos actually said — with clickable citations back to the exact moment in
each video. Everything is local except the chat model, which goes through OpenRouter (any model,
swappable).

![ytkb web chat](docs/ytkb.gif)

## What it does

- **Ingest a channel** — list its videos, fetch captions the cheap way (`yt-dlp`), and store them locally.
- **Strip the ads** — remove in-video sponsor segments (SponsorBlock, with an LLM fallback).
- **Ask questions** — a per-channel agent does keyword + semantic search over the transcripts and answers with citations (`video @ m:ss`).
- **Stay local** — embeddings, vector index, and storage all live on disk; only the answer model calls out.

## Features

- 📥 **Cheap ingestion** — captions via `yt-dlp` (creator subs › auto-captions › local Whisper fallback). No paid transcription.
- ✂️ **Ad removal** — SponsorBlock first; a cheap LLM pass for videos it doesn't cover.
- 🔎 **Agentic hybrid RAG** — the agent calls real tools: keyword search (SQLite FTS5), semantic search (LanceDB), read-transcript, list-videos. It searches, reads, and cites.
- 🔌 **Swappable models** — any OpenRouter model via `config.toml`; embeddings run locally (`fastembed`).
- 💬 **CLI + web chat** — a `kb` command line for sync/admin, plus a dark streaming web UI with Markdown, code blocks, and saved conversations.
- ♻️ **Incremental & resumable** — a per-video state machine tracks progress; `sync` only does new/failed work, `retry` re-runs failures, `reindex` rebuilds offline. Back up by copying a folder.

## Setup

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-..." > data/.env
cp data/config.toml.example data/config.toml   # optional: change models, chunking, language
```

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), `ffmpeg` (for the Whisper fallback), and an
[OpenRouter](https://openrouter.ai) API key. macOS/Linux.

## Usage

```bash
uv run kb add "https://www.youtube.com/@ycombinator" --no-shorts --name ycombinator
uv run kb sync ycombinator            # add --dry-run to preview
uv run kb status ycombinator
uv run kb ask ycombinator "How do I find a co-founder?"
uv run kb retry ycombinator           # re-process failed/no-transcript
uv run kb reindex ycombinator         # rebuild the index from local transcripts (e.g. after changing embedding_model)
uv run kb web                         # dark streaming chat UI at http://127.0.0.1:8000
```

The web chat streams answers token-by-token, renders Markdown (with syntax-highlighted code
blocks), and lets you pick the channel from the **+** button. It loads `marked`/`DOMPurify`/
`highlight.js` from CDN, so it needs network access (as does the agent itself).

Conversations are saved to `data/conversations.db`: the sidebar lists every chat (across all
channels), **New chat** starts a fresh one, and opening a past chat resumes it (follow-up
questions use the prior turns). A chat's channel is fixed when it's created.

### Long syncs
Backfilling a large channel can take a while (hundreds of videos). Keep the Mac awake for the
whole run — sleeping mid-sync can stall the network calls (yt-dlp, SponsorBlock) — by wrapping
it in `caffeinate`:
```bash
caffeinate -i uv run kb sync ycombinator
```

### Changing the embedding model
Edit `embedding_model` in `data/config.toml`, then run `uv run kb reindex <slug>`. This rebuilds
the vector index from the cleaned transcripts already on disk — no re-download, no LLM calls.
Switching embedding models requires a reindex because the vector dimension changes.

## How it works

**Ingestion** (per video, incremental, resumable):
`resolve channel → list videos → fetch captions (creator subs › auto › Whisper) → strip ads
(SponsorBlock › LLM) → chunk with timestamps → index into SQLite FTS5 + LanceDB`. Each video moves
through a state machine, so an interrupted sync resumes and only retries what failed.

**Querying:** a tool-using agent on OpenRouter searches the channel's index (keyword + semantic),
reads transcript context, and writes an answer with citations. Each channel is isolated under
`data/channels/<slug>/` (its own transcripts, SQLite DB, and vector table), so back up or move a
channel by copying its folder.

## Tech

Python 3.12 · `yt-dlp` · SponsorBlock · `faster-whisper` · `fastembed` (local embeddings) ·
LanceDB · SQLite + FTS5 · OpenRouter (via the `openai` SDK) · Typer (CLI) · FastAPI (web).
