# YouTube Channel Knowledge Base — Design Spec

**Date:** 2026-06-20
**Status:** Approved (pending final spec review)

## 1. Purpose

Build a local-first tool that ingests a YouTube channel's video transcripts, removes
in-video sponsor reads ("ads"), and exposes a per-channel **agentic RAG** assistant for
asking questions about that channel's expertise (e.g. YCombinator → startup advice).

Goals:
- Get transcripts as cheaply as possible (captions via `yt-dlp`, no paid transcription).
- Store everything locally; back up by copying a folder.
- Incrementally update a channel with new videos on demand, with full logging of what was
  analyzed, skipped, or failed, and a retry path for failures.
- One isolated "agent" per channel that answers from that channel's videos and cites them.

Non-goals (YAGNI): cloud hosting, multi-user accounts, redistribution of transcripts,
fine-tuning, real-time/live ingestion, cross-channel querying.

## 2. Key decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| LLM backend | **OpenRouter** (`openai` SDK, swappable model id) | One key, swap models freely; needs tool-calling-capable model |
| Embeddings | **Local** `fastembed` (`BAAI/bge-small-en-v1.5`) | OpenRouter has no embeddings; free, fast, offline |
| Ad removal | **SponsorBlock API → cheap LLM fallback** | Free segments first; LLM only when SponsorBlock has no data |
| No-caption fallback | **faster-whisper** (local), **on by default**, per-channel toggle | Free, uses the Mac; slow, so non-blocking and resumable |
| Retrieval | **Agentic RAG, hybrid toolbox** | Better on comparative/multi-hop questions; degrades to ~classic RAG cost on easy ones via early-stop |
| Ingest scope | **Per-channel filters at `add` time** | User sets count/date/min-length/shorts per channel |
| Interface | **CLI + local web chat** | CLI for sync/admin; web for day-to-day asking |
| Vector store | **LanceDB** (embedded, one dir per channel) | No server, file-based, easy backup |
| State/logs | **SQLite (+FTS5)** per channel | Per-video state machine, run logs, and lexical search index |
| Channel isolation | **One folder per channel** | Retrieval never bleeds across channels; delete/rebuild one in isolation |

## 3. Architecture overview

Two halves over a shared core library:

```
                ┌─────────────── core (Python package) ───────────────┐
                │  channel resolver · ingestion pipeline · indexes ·   │
                │  retrieval tools · agent loop · state/db · config    │
                └──────────────────────────────────────────────────────┘
   CLI (Typer) ─────────────┘                          └───────── Web (FastAPI + static page)
```

The CLI and web UI are thin layers; all logic lives in the core so both share one code path.

## 4. Data layout (all local)

```
data/
  config.toml                 # defaults: model ids, chunk size/overlap, embedding model, langs
  .env                        # OPENROUTER_API_KEY
  channels/
    <slug>/                   # e.g. ycombinator
      channel.json            # channel id, handle, title, filters, model overrides, last_sync
      videos.db               # SQLite: videos (state machine), runs (log), chunks (+FTS5)
      transcripts/
        <video_id>.raw.json   # original timed captions (segments with start/end/text)
        <video_id>.clean.txt  # ad-stripped plain text (with light timestamp anchors)
      vectors/                # LanceDB table for this channel's chunk embeddings
```

Notes:
- A channel is fully self-contained under its folder. Channels are discovered by scanning
  `data/channels/` (no separate registry file). Removing a channel = deleting its dir.
- `config.toml` holds global defaults; `channel.json` can override model ids and filters.

## 5. Ingestion pipeline (per video)

Order of operations:

1. **Resolve channel** (once per channel): accept any URL form — `@handle`, `/c/<name>`,
   `/user/<name>`, `/channel/<id>`, or even a single video URL → derive the owning channel.
   Persist canonical channel id + handle + title in `channel.json`.
2. **List videos**: enumerate the channel's uploads via `yt-dlp` (flat/fast listing), apply
   per-channel filters (`--max`, `--since`, `--min-length`, `--no-shorts`). Shorts detected by
   duration threshold (default < 60s) when `--no-shorts` is set.
3. **Fetch transcript** (caption priority): creator-uploaded subtitles › YouTube auto-captions
   › **faster-whisper** on downloaded audio (default on; per-channel toggle). Store timed segments to
   `<video_id>.raw.json`. Language defaults to English, configurable; Whisper auto-detects.
4. **Strip ads**: query SponsorBlock for the video's sponsor/selfpromo segments; map those time
   ranges onto caption segments and drop them. If SponsorBlock returns nothing, run a cheap LLM
   pass to detect/remove sponsor reads. Write result to `<video_id>.clean.txt`.
5. **Chunk + index**: split the clean transcript into time-windowed chunks (~500 tokens,
   small overlap), preserving each chunk's start timestamp + video id. Insert chunks into
   SQLite (`chunks` table + FTS5 virtual table) and embed them into the channel's LanceDB table.
6. Mark video `DONE`.

Idempotency/resumability: each step writes atomically and updates the video's state, so an
interrupted `sync` resumes cleanly and never reprocesses completed work.

## 6. State machine & logging

`videos` table tracks per-video status:

```
discovered
  → transcript_fetched | whisper_transcribed
  → ad_stripped
  → indexed  (== DONE)

Failure / terminal-but-retryable states (store last error + attempt_count + last_attempt_at):
  failed_fetch      — listing/caption fetch error (network, throttling)
  no_transcript     — no captions and Whisper disabled/failed (captions may appear later)
  failed_embed      — chunking/embedding/index error
```

`runs` table logs each `sync`/`retry` invocation (start/end, counts: new, done, skipped,
failed) for an audit trail. `kb status` summarizes counts by state, last sync time, and
failures. `kb retry` re-queues `failed_*` and `no_transcript` videos.

## 7. Query: agentic RAG

Per channel, an agent on OpenRouter with a channel-scoped system prompt ("You answer only
from <channel>'s videos; cite video title + timestamp; say so if the videos don't cover it").

Tools (all local, free to run):

| Tool | Backed by | Purpose |
|---|---|---|
| `keyword_search(query, k)` | SQLite FTS5 (BM25) | exact terms, names, jargon — ranked "grep" |
| `semantic_search(query, k)` | fastembed + LanceDB | paraphrases, concepts |
| `read_transcript(video_id, around_ts?, window?)` | `.clean.txt` on disk | full context around a hit |
| `list_videos(filter)` / `search_titles(query)` | SQLite metadata | scope by topic/date; "which videos cover X" |

Agent loop: think → call tool(s) → read results → optionally search again → synthesize answer
with citations. **Early-stop**: simple questions resolve in one search, keeping cost near
classic-RAG levels; cost grows only for genuinely hard/multi-hop questions. Citations render as
`https://youtu.be/<id>?t=<seconds>` deep links plus the video title.

Model: default to a cheap tool-calling-capable model on OpenRouter (Haiku / Gemini Flash
class); overridable globally or per channel in config.

## 8. Commands (CLI — Typer)

```
kb add <url> [--max N] [--since YYYY-MM-DD] [--min-length SEC] [--no-shorts]
             [--whisper/--no-whisper] [--name <slug>]
   → resolve channel, save channel.json with filters. Does not auto-sync.

kb sync [<slug> | --all] [--dry-run]
   → discover new videos + process new/failed ones incrementally.
     --dry-run prints what would be processed (and a rough LLM-cost estimate) without doing it.

kb ask <slug> "question"      → agentic answer + cited sources, printed to terminal.
kb status [<slug>]            → per-state counts, last sync, failures.
kb retry <slug>               → re-process failed_* and no_transcript videos.
kb list                       → channels overview (slug, title, #videos, last sync).
kb web                        → launch local web chat (channel dropdown + streaming chat).
kb remove <slug>              → delete a channel's folder/data (with confirmation).
```

## 9. Web UI

FastAPI serving a single static page: a channel dropdown + a chat box. Asking streams the
agent's answer (SSE) and renders citations as clickable timestamped YouTube links. No auth,
binds to localhost only. Same core `ask` path as the CLI.

## 10. Configuration

- `.env`: `OPENROUTER_API_KEY`.
- `config.toml` defaults: chat model id, ad-strip model id, embedding model, chunk size/overlap,
  default language(s), Whisper on/off + model size, request delays/backoff, top-K defaults.
- `channel.json` per channel: canonical id/handle/title, filters, optional model overrides,
  `last_sync`, Whisper toggle.

## 11. Cross-cutting concerns ("other points")

- **Channel resolution** from any URL form, including a single-video URL → owning channel.
- **Rate-limit safety**: polite inter-request delays + exponential backoff on `yt-dlp`;
  optional cookies file to pass if YouTube throttles.
- **Cost controls**: cheap default models; `--dry-run` preview; LLM ad-strip only when
  SponsorBlock is empty; local embeddings so retrieval has zero ongoing cost.
- **Re-sync hygiene**: detect deleted/private videos on re-list; dedup strictly by `video_id`.
- **Language**: English captions by default, configurable; Whisper auto-detects.
- **Portability/backup**: copy `data/` (or a single channel folder) to back up or move.
- **ToS / privacy**: personal-use transcript archiving for local research; not for
  redistribution. Respect YouTube ToS; throttle politely.

## 12. Testing strategy

- **Unit**: channel-URL resolution (all forms); SponsorBlock segment→caption mapping; chunking
  (boundaries, timestamps, overlap); state-machine transitions; filter logic (shorts/date/length).
- **Integration (mocked network)**: ingestion pipeline end-to-end against a fixture caption file
  and a fixture SponsorBlock response; incremental sync only touches new/failed videos; retry
  re-queues the right states.
- **Retrieval/agent**: tools return expected hits over a small fixture corpus; agent loop with a
  stubbed LLM (deterministic tool calls) produces a cited answer; early-stop on a simple query.
- **Manual smoke**: run `add` + `sync` against one small real channel; `ask` a few questions;
  verify citations deep-link correctly.

## 13. Build sequence (high level — detailed in the implementation plan)

1. Project scaffold (`uv`), config loading, data-dir + per-channel layout, SQLite schema.
2. Channel resolution + video listing + filters.
3. Transcript fetch (captions) + raw storage; Whisper fallback (optional).
4. Ad removal (SponsorBlock + LLM fallback) + clean storage.
5. Chunking + FTS5 + LanceDB indexing; state machine + run logging.
6. `sync` / `retry` / `status` / `list` / `add` / `remove` CLI.
7. Retrieval tools + agentic answer loop + citations; `ask` CLI.
8. Web chat (FastAPI + static page, streaming).
9. Tests throughout; manual smoke against a real channel.
