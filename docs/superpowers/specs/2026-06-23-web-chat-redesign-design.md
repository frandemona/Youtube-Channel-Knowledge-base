# Web Chat Redesign + Streaming — Design Spec

**Date:** 2026-06-23
**Status:** Approved (pending final spec review)
**Builds on:** [2026-06-20-youtube-channel-knowledgebase-design.md](2026-06-20-youtube-channel-knowledgebase-design.md) §9 (web UI)

## 1. Purpose

Replace the bare white `kb web` chat page with a polished dark chat UI (modeled on the
provided AetherAI reference), render the agent's Markdown answers properly, and stream
responses token-by-token with live status during the search phase so it feels fast.

Non-goals (YAGNI): auth, multi-user, message persistence/history across reloads, model
picker, file uploads, voice, mobile-first layout (desktop-first is fine; it should not break
on mobile but won't be optimized).

## 2. Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Front-end deps | **CDN** (`marked`, `DOMPurify`, `highlight.js` + a dark theme CSS), pinned versions | The app already requires network (OpenRouter), so offline buys nothing; no vendoring/build |
| Streaming transport | **SSE** over `POST /api/ask/stream` (`text/event-stream`), read client-side with `fetch()` + stream reader | EventSource is GET-only; fetch-stream handles POST bodies and is well-supported |
| Streaming mechanism | Agent **streams every model call** and forwards answer tokens live; tool-call steps emit status, not text | True token streaming, no wasted "regenerate" call |
| Markdown | Client-side `marked` → **`DOMPurify` sanitize** → innerHTML, re-rendered on each token | LLM output is untrusted HTML; sanitizing is mandatory |
| Code blocks | `highlight.js` + a header bar (language label + **Copy code** button) | Matches the reference conversation view |
| Page structure | Single no-build page split into `index.html` + `static/styles.css` + `static/app.js`, served via FastAPI `StaticFiles` | Maintainable; still zero build step |
| Channel selection | A **`+` button opens a channel-picker popover** (from `/api/channels`); replaces the old `<select>` | Matches the reference's `+` affordance |
| CLI | `agent.answer()` (non-streaming) is **unchanged**; streaming is additive for the web only | Don't disturb the tested CLI path |

## 3. Visual design

Dark theme (near-black background `#0a0a0a`-ish, soft 1px borders, muted gray text, a subtle
accent). Two states inside one page:

**Empty state** (centered hero):
- Title "Channel Knowledge Base" + subtitle "Ask the <channel title> agent anything."
- A rounded input bar: `+` (channel picker) on the left, a growing textarea, a send (↑) button on the right.
- Below the bar, **example-prompt chips**: "How do I find a co-founder?", "When should I launch?",
  "How do I talk to users?", and a "…" chip (the chips fill the input and send on click).
- A small muted footer line.

**Conversation state** (after first send): hero collapses; a scrollable message list fills the
space; the input bar sits at the bottom.
- **User message**: right-aligned pill, light text on a dark rounded bubble.
- **Assistant message**: left-aligned block (wider), rendered Markdown.
  - **Status lines** appear first (muted, e.g. "Searching transcripts…", "Reading 2 videos…")
    and are replaced/cleared once tokens begin.
  - **Code blocks** render as a dark card: header row with the language label + a "Copy code"
    button (copies the raw code), syntax-highlighted body.
  - **Citations** render at the end as a "Sources" list of clickable links
    (`<video title> @ <m:ss>` → `https://youtu.be/<id>?t=<sec>`), opening in a new tab.

**Channel picker**: clicking `+` opens a small popover listing channel slugs; selecting one sets
the active channel (shown as a chip next to `+`). Defaults to the first channel on load.

## 4. Streaming protocol

`POST /api/ask/stream` with JSON body `{"slug": str, "question": str}` returns
`text/event-stream`. Each event is a line `data: <json>\n\n` where `<json>` is one of:

| `type` | Fields | When |
|---|---|---|
| `status` | `text` | During the tool/search phase (one per tool action) |
| `token` | `text` | Each answer token as it streams |
| `citations` | `citations: [{video_id,title,start,url}]` | After the answer, before `done` |
| `done` | — | Stream complete |
| `error` | `text` | No API key, or any agent/network error (then the stream ends) |

The browser reads the body stream, splits on `\n\n`, parses each `data:` payload, and:
appends `status` lines to a status area; appends `token` text to the answer buffer and
re-renders Markdown; renders `citations`; closes on `done`/`error`.

## 5. Backend changes

### 5.1 `ytkb/llm.py` — add streaming
`LLMClient.stream_with_tools(messages, model, tools) -> Iterator[StreamEvent]` where iterating
yields content-delta strings, and accumulated tool calls are exposed after iteration. Concretely:
the method calls `chat.completions.create(..., tools=tools, tool_choice="auto", stream=True)` and
yields `("token", delta_text)` for each non-empty `choices[0].delta.content`, while accumulating
`choices[0].delta.tool_calls` (by index: `id`, `function.name`, `function.arguments` concatenated).
After the generator is exhausted it exposes the final assembled `tool_calls` list (empty if none).
Implementation shape: a small `ToolCallStream` object that is iterable (yields token strings) and
has a `.tool_calls` attribute populated once iteration completes. `client` stays injectable so
tests pass a fake that emits scripted chunks.

### 5.2 `ytkb/agent.py` — add `answer_stream`
`answer_stream(question, channel_title, store, llm, *, chat_model, top_k, max_steps=5) -> Iterator[dict]`
yields the protocol events from §4. It reuses the existing system prompt, `build_tools`, and
`_collect_citations`. Loop:
- For each step, open `llm.stream_with_tools(messages, chat_model, specs)`. Forward each token
  string as `{"type":"token","text":...}` and accumulate them into an assistant content buffer.
- When the stream ends, read `.tool_calls`. If present: append the assistant tool-call message,
  emit a `{"type":"status","text":...}` per call (human-readable: "Searching transcripts for
  '<query>'…", "Reading <video_id>…", "Listing videos…"), run `dispatch`, append `role:"tool"`
  results, collect citations, and continue the loop (do not emit the next step's nothing).
- If no tool_calls: the streamed tokens were the answer — break.
- After the loop, emit `{"type":"citations","citations":[…]}` (deduped, from tool results actually
  used) then `{"type":"done"}`.
- Wrap the whole body so any exception emits `{"type":"error","text":str(e)}` then `done`.

`answer()` (non-streaming, used by the CLI) is left exactly as-is.

### 5.3 `ytkb/web/app.py`
- Mount static assets: `app.mount("/static", StaticFiles(directory=STATIC), name="static")`.
- `GET /` serves `static/index.html` (unchanged behavior).
- `GET /api/channels` unchanged.
- Keep `POST /api/ask` (non-streaming) for the existing tests / fallback.
- Add `POST /api/ask/stream`: build context; if `ctx.llm is None`, return a stream that emits a
  single `error` event; else return `StreamingResponse(_sse(answer_stream(...)),
  media_type="text/event-stream")` where `_sse` wraps each event dict as `data: {json}\n\n`.

## 6. Frontend structure

- `static/index.html`: head loads pinned CDN `<script>`s (`marked`, `dompurify`, `highlight.js`)
  and the highlight.js dark theme `<link>`, plus our `styles.css` and `app.js`. Body has the hero,
  the message list, the input bar, the channel-picker popover, and the example chips.
- `static/styles.css`: the dark theme, both states, message bubbles, code-block card, popover, chips.
- `static/app.js`: load channels → populate picker; chip clicks; send handler that POSTs to
  `/api/ask/stream`, reads the stream, drives status/token/citations/done/error; Markdown render
  (`DOMPurify.sanitize(marked.parse(buffer))`); post-render pass to highlight code blocks and inject
  the language label + Copy-code button; autoscroll.
- CDN versions are pinned (exact version in the `<script src>` URLs) for reproducibility.

## 7. Error handling

- No `OPENROUTER_API_KEY`: the stream emits one `error` event ("No OPENROUTER_API_KEY configured in
  data/.env"); the UI shows it inline in the assistant area.
- Mid-stream agent/network error: an `error` event is emitted and the stream closes; the UI stops the
  typing state and shows the message.
- Empty question or no channel selected: the client refuses to send (no request).

## 8. Testing

- **`llm.stream_with_tools`**: a fake client emits scripted streaming chunks (content deltas; and a
  separate case with tool-call deltas across chunks). Assert the yielded token strings and that
  `.tool_calls` is assembled correctly (name + concatenated arguments).
- **`agent.answer_stream`**: a scripted fake LLM does one tool step (tool-call deltas) then a content
  stream. Assert the event sequence: `status` → `token`…`token` → `citations` (with the right
  `youtu.be/<id>?t=<sec>` URL) → `done`. A second test: `llm is None`/exception path yields an
  `error` event.
- **Web endpoint**: FastAPI `TestClient` reads `POST /api/ask/stream` (monkeypatched
  `agent.answer_stream` yielding a fixed event list) and asserts the SSE body contains the events in
  order; existing `/api/channels` and non-streaming `/api/ask` tests stay green.
- **Frontend**: manual verification — launch `kb web`, confirm empty state, channel picker, a chip
  send, live status, streamed Markdown answer with a code block + Copy button, and clickable citations.

## 9. Out of scope / follow-ups
History persistence, multiple concurrent conversations, model selection in the UI, mobile-optimized
layout, and the previously-deferred items (`last_sync`, SSE for nothing else, etc.).
