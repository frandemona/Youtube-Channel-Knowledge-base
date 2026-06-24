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
uv run kb reindex ycombinator         # rebuild the index from local transcripts (e.g. after changing embedding_model)
uv run kb web                         # dark streaming chat UI at http://127.0.0.1:8000
```

The web chat streams answers token-by-token, renders Markdown (with syntax-highlighted code
blocks), and lets you pick the channel from the **+** button. It loads `marked`/`DOMPurify`/
`highlight.js` from CDN, so it needs network access (as does the agent itself).

### Changing the embedding model
Edit `embedding_model` in `data/config.toml`, then run `uv run kb reindex <slug>`. This rebuilds
the vector index from the cleaned transcripts already on disk — no re-download, no LLM calls.
Switching embedding models requires a reindex because the vector dimension changes.

Data lives under `data/channels/<slug>/`. Back up by copying `data/`.
