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
