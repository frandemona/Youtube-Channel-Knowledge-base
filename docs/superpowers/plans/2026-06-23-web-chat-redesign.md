# Web Chat Redesign + Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bare `kb web` chat page with a dark, streaming chat UI that renders the agent's Markdown answers, shows live search status, and lets the user pick a channel from a `+` button.

**Architecture:** Backend gains token streaming: `LLMClient.stream_with_tools` streams a completion (accumulating tool-call deltas), `agent.answer_stream` yields typed events (status/token/citations/done/error), and `POST /api/ask/stream` serves them as SSE. Frontend is a single no-build page (`index.html` + `static/styles.css` + `static/app.js`) served via FastAPI `StaticFiles`, using CDN `marked`/`DOMPurify`/`highlight.js`.

**Tech Stack:** Existing `ytkb` stack (Python 3.12, FastAPI, openai SDK for OpenRouter); CDN `marked@12`, `dompurify@3`, `highlight.js@11`. No new Python deps.

## Global Constraints

- Python **3.12**; run with `uv` (`uv run pytest`, `uv run kb web`).
- The non-streaming `agent.answer()` and `POST /api/ask` are **unchanged** (CLI + existing tests depend on them). Streaming is additive.
- SSE event protocol — each event is a dict with a `type` field, one of: `status` (`text`), `token` (`text`), `citations` (`citations: list[dict]`), `done`, `error` (`text`). Over the wire each is `data: <json>\n\n`.
- Front-end libraries load from **CDN, pinned versions**: `marked@12.0.2`, `dompurify@3.1.6`, `highlight.js@11.9.0` (+ its `github-dark` theme CSS). No vendoring, no build step.
- Markdown is **always sanitized** (`DOMPurify.sanitize(marked.parse(text))`) before insertion — LLM output is untrusted.
- Tests must not hit the network/LLM: inject fakes (fake openai client; scripted fake LLM; monkeypatched `agent.answer_stream`).
- TDD per task: failing test → see it fail → implement → see it pass → commit.

---

## File Structure

```
ytkb/llm.py        # + ToolCallStream, LLMClient.stream_with_tools
ytkb/agent.py      # + answer_stream(), _status_for()  (answer() untouched)
ytkb/web/app.py    # + StaticFiles mount, POST /api/ask/stream, _sse helper
ytkb/web/static/index.html   # rewritten: dark UI shell + CDN libs
ytkb/web/static/styles.css   # new: dark theme
ytkb/web/static/app.js       # new: channel picker, chips, streaming client, markdown render
tests/test_llm_stream.py     # ToolCallStream / stream_with_tools
tests/test_agent_stream.py   # answer_stream events
tests/test_web.py            # + streaming-endpoint tests (existing tests kept)
```

---

### Task 1: `LLMClient.stream_with_tools` (streaming + tool-call accumulation)

**Files:**
- Modify: `ytkb/llm.py`
- Test: `tests/test_llm_stream.py`

**Interfaces:**
- Consumes: an injectable `client` whose `chat.completions.create(..., stream=True)` returns an iterable of chunks shaped like the OpenAI streaming API (`chunk.choices[0].delta.content`, `chunk.choices[0].delta.tool_calls[i].{index,id,function.name,function.arguments}`).
- Produces:
  - `ytkb.llm.ToolCallStream(raw_stream)` — iterating it yields each non-empty content delta string; after iteration completes, `.tool_calls` is a list of `ToolCall` objects.
  - `ytkb.llm.ToolCall` with `.id: str | None` and `.function` (an object with `.name: str` and `.arguments: str`).
  - `LLMClient.stream_with_tools(messages: list[dict], model: str, tools: list[dict]) -> ToolCallStream`.

- [ ] **Step 1: Write the failing test**

`tests/test_llm_stream.py`:
```python
from types import SimpleNamespace
from ytkb.llm import LLMClient, ToolCallStream


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tc(index, id=None, name=None, args=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=args))


class FakeStreamingClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured = {}

    class _Chat:
        def __init__(self, outer): self.outer = outer
        @property
        def completions(self): return FakeStreamingClient._Comp(self.outer)

    class _Comp:
        def __init__(self, outer): self.outer = outer
        def create(self, model, messages, tools, tool_choice, stream):
            self.outer.captured = {"stream": stream, "tool_choice": tool_choice}
            return iter(self.outer._chunks)

    @property
    def chat(self): return FakeStreamingClient._Chat(self)


def test_stream_yields_content_deltas():
    chunks = [_chunk(content="Hello "), _chunk(content="world")]
    llm = LLMClient(api_key="k", client=FakeStreamingClient(chunks))
    stream = llm.stream_with_tools([{"role": "user", "content": "x"}], model="m", tools=[])
    assert list(stream) == ["Hello ", "world"]
    assert stream.tool_calls == []


def test_stream_accumulates_tool_calls_across_chunks():
    chunks = [
        _chunk(tool_calls=[_tc(0, id="call1", name="keyword_search", args='{"que')]),
        _chunk(tool_calls=[_tc(0, args='ry":"cofounder"}')]),
    ]
    llm = LLMClient(api_key="k", client=FakeStreamingClient(chunks))
    stream = llm.stream_with_tools([], model="m", tools=[{"x": 1}])
    assert list(stream) == []  # no content
    tc = stream.tool_calls[0]
    assert tc.id == "call1"
    assert tc.function.name == "keyword_search"
    assert tc.function.arguments == '{"query":"cofounder"}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_stream.py -v`
Expected: FAIL with `ImportError: cannot import name 'ToolCallStream'`

- [ ] **Step 3: Implement in `ytkb/llm.py`**

Add (keep the existing `complete`/`chat_with_tools` methods unchanged):
```python
from dataclasses import dataclass


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str | None
    function: _Fn


class ToolCallStream:
    """Iterate to get content-delta strings; after iteration, `.tool_calls` is populated."""

    def __init__(self, raw_stream):
        self._raw = raw_stream
        self.tool_calls: list[ToolCall] = []

    def __iter__(self):
        acc: dict[int, dict] = {}
        for chunk in self._raw:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = acc.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
        self.tool_calls = [
            ToolCall(id=s["id"], function=_Fn(name=s["name"], arguments=s["args"]))
            for _, s in sorted(acc.items())
        ]
```
And add the method to `LLMClient`:
```python
    def stream_with_tools(self, messages: list[dict], model: str, tools: list[dict]) -> "ToolCallStream":
        raw = self._client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto", stream=True
        )
        return ToolCallStream(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_stream.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/llm.py tests/test_llm_stream.py
git commit -m "feat: LLMClient.stream_with_tools with tool-call delta accumulation"
```

---

### Task 2: `agent.answer_stream` (event generator)

**Files:**
- Modify: `ytkb/agent.py`
- Test: `tests/test_agent_stream.py`

**Interfaces:**
- Consumes: `LLMClient.stream_with_tools(messages, model, tools) -> ToolCallStream` (Task 1); `build_tools`, `_collect_citations` (existing); `models.Citation`.
- Produces:
  - `ytkb.agent._status_for(name: str, args_json: str) -> str` (human-readable status line for a tool call).
  - `ytkb.agent.answer_stream(question, channel_title, store, llm, *, chat_model, top_k, max_steps=5) -> Iterator[dict]` yielding the protocol events (`status`/`token`/`citations`/`done`/`error`).

- [ ] **Step 1: Write the failing test**

`tests/test_agent_stream.py`:
```python
from types import SimpleNamespace
from ytkb.agent import answer_stream
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


class _Stream:
    def __init__(self, tokens, tool_calls):
        self._tokens = tokens
        self.tool_calls = tool_calls

    def __iter__(self):
        return iter(self._tokens)


class ScriptedLLM:
    """Step 1 -> a keyword_search tool call (no content); step 2 -> streamed answer."""
    def __init__(self):
        self.n = 0

    def stream_with_tools(self, messages, model, tools):
        self.n += 1
        if self.n == 1:
            tc = SimpleNamespace(
                id="c1", function=SimpleNamespace(name="keyword_search", arguments='{"query":"cofounder"}'))
            return _Stream([], [tc])
        return _Stream(["Find ", "a technical cofounder."], [])


def test_answer_stream_emits_status_tokens_citations_done():
    events = list(answer_stream("How?", "Y Combinator", FakeStore(), ScriptedLLM(),
                                chat_model="m", top_k=5))
    types = [e["type"] for e in events]
    assert types == ["status", "token", "token", "citations", "done"]
    assert "transcripts" in events[0]["text"].lower()
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Find a technical cofounder."
    assert events[3]["citations"][0]["url"] == "https://youtu.be/vid123?t=42"


def test_answer_stream_emits_error_on_exception():
    class BoomLLM:
        def stream_with_tools(self, messages, model, tools):
            raise RuntimeError("boom")

    events = list(answer_stream("q", "C", FakeStore(), BoomLLM(), chat_model="m", top_k=5))
    assert events[0]["type"] == "error" and "boom" in events[0]["text"]
    assert events[-1]["type"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_stream.py -v`
Expected: FAIL with `ImportError: cannot import name 'answer_stream'`

- [ ] **Step 3: Implement in `ytkb/agent.py`**

Add these imports at the top (next to the existing `import json`):
```python
from dataclasses import asdict
```
Then add (leave `answer()` and `_collect_citations()` exactly as they are):
```python
def _status_for(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json or "{}")
    except (json.JSONDecodeError, TypeError):
        args = {}
    if name in ("keyword_search", "semantic_search"):
        q = args.get("query", "")
        return f"Searching transcripts for '{q}'…" if q else "Searching transcripts…"
    if name == "read_transcript":
        return f"Reading {args.get('video_id', 'a video')}…"
    if name == "list_videos":
        return "Listing videos…"
    return "Working…"


def answer_stream(question, channel_title, store, llm, *, chat_model, top_k, max_steps=5):
    try:
        specs, dispatch = build_tools(store, top_k)
        messages = [
            {"role": "system", "content": SYSTEM.format(channel=channel_title)},
            {"role": "user", "content": question},
        ]
        cited: dict[str, Citation] = {}

        for _ in range(max_steps):
            stream = llm.stream_with_tools(messages, chat_model, specs)
            content = ""
            for token in stream:
                content += token
                yield {"type": "token", "text": token}
            tool_calls = stream.tool_calls
            if not tool_calls:
                break
            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                yield {"type": "status", "text": _status_for(tc.function.name, tc.function.arguments)}
                args = json.loads(tc.function.arguments or "{}")
                result = dispatch(tc.function.name, args)
                _collect_citations(tc.function.name, result, cited)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        yield {"type": "citations", "citations": [asdict(c) for c in cited.values()]}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "text": str(e)}
        yield {"type": "done"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_stream.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/agent.py tests/test_agent_stream.py
git commit -m "feat: agent.answer_stream event generator (status/token/citations/done/error)"
```

---

### Task 3: Streaming endpoint + static mount in `web/app.py`

**Files:**
- Modify: `ytkb/web/app.py`
- Test: `tests/test_web.py` (add tests; keep existing ones)

**Interfaces:**
- Consumes: `agent.answer_stream` (Task 2); `sync.load_channel`, `sync.build_context`, `load_config`.
- Produces: `POST /api/ask/stream` returning `text/event-stream`; a `/static` mount serving `ytkb/web/static/`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web.py`:
```python
def test_ask_stream_emits_sse_events(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))

    class FakeCtx:
        store = object()
        llm = object()

    from ytkb.channel import ChannelInfo, ChannelFilters
    info = ChannelInfo("UC1", "@yc", "Y Combinator", "u")
    monkeypatch.setattr(webapp.sync, "build_context", lambda cfg, slug: FakeCtx())
    monkeypatch.setattr(webapp.sync, "load_channel", lambda cfg, slug: (info, ChannelFilters()))
    monkeypatch.setattr(webapp.agent, "answer_stream", lambda *a, **k: iter([
        {"type": "status", "text": "Searching transcripts…"},
        {"type": "token", "text": "Hello"},
        {"type": "citations", "citations": [{"video_id": "v1", "title": "T", "start": 1.0,
                                             "url": "https://youtu.be/v1?t=1"}]},
        {"type": "done"},
    ]))
    client = TestClient(create_app())
    r = client.post("/api/ask/stream", json={"slug": "yc", "question": "how?"})
    assert r.status_code == 200
    body = r.text
    assert 'data: {"type": "status"' in body
    assert '"text": "Hello"' in body
    assert "youtu.be/v1?t=1" in body
    assert '"type": "done"' in body


def test_ask_stream_no_api_key_emits_error(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))

    class NoLLMCtx:
        store = object()
        llm = None

    from ytkb.channel import ChannelInfo, ChannelFilters
    info = ChannelInfo("UC1", "@yc", "Y Combinator", "u")
    monkeypatch.setattr(webapp.sync, "build_context", lambda cfg, slug: NoLLMCtx())
    monkeypatch.setattr(webapp.sync, "load_channel", lambda cfg, slug: (info, ChannelFilters()))
    client = TestClient(create_app())
    r = client.post("/api/ask/stream", json={"slug": "yc", "question": "how?"})
    assert r.status_code == 200
    assert '"type": "error"' in r.text
    assert "OPENROUTER_API_KEY" in r.text
```
(`tests/test_web.py` already imports `from fastapi.testclient import TestClient`, `from ytkb.web.app import create_app`, and `import ytkb.web.app as webapp`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web.py::test_ask_stream_emits_sse_events -v`
Expected: FAIL (404 / route not found)

- [ ] **Step 3: Implement in `ytkb/web/app.py`**

Update imports and `create_app`:
```python
import json
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import sync, agent
from ..config import load_config

STATIC = Path(__file__).parent / "static"


class AskReq(BaseModel):
    slug: str
    question: str


def _sse(events):
    for ev in events:
        yield f"data: {json.dumps(ev)}\n\n"


def create_app(cfg=None) -> FastAPI:
    app = FastAPI(title="ytkb")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

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

    @app.post("/api/ask/stream")
    def ask_stream(req: AskReq):
        c = cfg or load_config()
        info, _ = sync.load_channel(c, req.slug)
        ctx = sync.build_context(c, req.slug)
        if ctx.llm is None:
            events = iter([
                {"type": "error", "text": "No OPENROUTER_API_KEY configured in data/.env"},
                {"type": "done"},
            ])
        else:
            events = agent.answer_stream(req.question, info.title, ctx.store, ctx.llm,
                                         chat_model=c.chat_model, top_k=c.top_k)
        return StreamingResponse(_sse(events), media_type="text/event-stream")

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web.py -v`
Expected: PASS (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add ytkb/web/app.py tests/test_web.py
git commit -m "feat: SSE streaming endpoint /api/ask/stream and static mount"
```

---

### Task 4: Dark streaming chat frontend

**Files:**
- Modify: `ytkb/web/static/index.html` (full rewrite)
- Create: `ytkb/web/static/styles.css`, `ytkb/web/static/app.js`
- Test: add one served-page assertion to `tests/test_web.py`; the rest is manual verification.

**Interfaces:**
- Consumes: `GET /api/channels`, `POST /api/ask/stream` (SSE), `/static/*` (Task 3).
- Produces: the chat UI. No Python interface.

> **Implementer note:** the JS below is the required behavior — implement it as written. Use the `frontend-design` skill to refine `styles.css` so the dark theme matches the AetherAI reference (near-black background, soft borders, rounded input bar, pill messages, code-block cards). Keep the element ids/classes the JS depends on (`#channel-btn`, `#channel-menu`, `#messages`, `#hero`, `#q`, `#send`, `.chip`, `[data-slug]`).

- [ ] **Step 1: Write the failing served-page test**

Add to `tests/test_web.py`:
```python
def test_index_has_streaming_ui(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())
    html = client.get("/").text
    assert 'id="messages"' in html and 'id="channel-btn"' in html
    assert "/static/app.js" in html
    # static asset is served
    assert client.get("/static/app.js").status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web.py::test_index_has_streaming_ui -v`
Expected: FAIL (old index.html lacks these ids; `/static/app.js` 404)

- [ ] **Step 3: Write `ytkb/web/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Channel Knowledge Base</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"/>
  <link rel="stylesheet" href="/static/styles.css"/>
  <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
</head>
<body>
  <main id="app">
    <div id="hero">
      <h1>Channel Knowledge Base</h1>
      <p id="subtitle">Ask the channel's agent anything.</p>
    </div>

    <div id="messages"></div>

    <div id="composer">
      <div id="bar">
        <div id="channel-wrap">
          <button id="channel-btn" type="button" title="Select channel">+</button>
          <span id="channel-label"></span>
          <div id="channel-menu" hidden></div>
        </div>
        <textarea id="q" rows="1" placeholder="Ask anything …"></textarea>
        <button id="send" type="button" title="Send">↑</button>
      </div>
      <div id="chips">
        <button class="chip">How do I find a co-founder?</button>
        <button class="chip">When should I launch?</button>
        <button class="chip">How do I talk to users?</button>
      </div>
    </div>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write `ytkb/web/static/app.js`** (required behavior — implement as written)

```javascript
const $ = (s) => document.querySelector(s);
let currentSlug = null;

function md(text) {
  return DOMPurify.sanitize(marked.parse(text || ""));
}

function enhanceCode(container) {
  container.querySelectorAll("pre > code").forEach((code) => {
    if (code.dataset.enhanced) return;
    code.dataset.enhanced = "1";
    try { hljs.highlightElement(code); } catch (e) {}
    const pre = code.parentElement;
    const lang = (code.className.match(/language-(\w+)/) || [, "code"])[1];
    const header = document.createElement("div");
    header.className = "code-header";
    header.innerHTML = `<span>${lang}</span><button class="copy">Copy code</button>`;
    header.querySelector(".copy").onclick = () => {
      navigator.clipboard.writeText(code.innerText);
      header.querySelector(".copy").textContent = "Copied";
      setTimeout(() => (header.querySelector(".copy").textContent = "Copy code"), 1200);
    };
    pre.prepend(header);
  });
}

async function loadChannels() {
  const d = await (await fetch("/api/channels")).json();
  const menu = $("#channel-menu");
  menu.innerHTML = (d.channels || [])
    .map((c) => `<button data-slug="${c}">${c}</button>`)
    .join("");
  menu.querySelectorAll("[data-slug]").forEach((b) => {
    b.onclick = () => { setChannel(b.dataset.slug); menu.hidden = true; };
  });
  if (d.channels && d.channels.length) setChannel(d.channels[0]);
}

function setChannel(slug) {
  currentSlug = slug;
  $("#channel-label").textContent = slug;
}

$("#channel-btn").onclick = () => { $("#channel-menu").hidden = !$("#channel-menu").hidden; };

function addMessage(role) {
  document.body.classList.add("chatting");
  const el = document.createElement("div");
  el.className = "msg " + role;
  $("#messages").appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

async function ask(question) {
  if (!question.trim() || !currentSlug) return;
  addMessage("user").textContent = question;
  const bot = addMessage("assistant");
  const status = document.createElement("div");
  status.className = "status";
  const body = document.createElement("div");
  body.className = "body";
  bot.append(status, body);

  let answer = "";
  let res;
  try {
    res = await fetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug: currentSlug, question }),
    });
  } catch (e) { status.textContent = "Network error: " + e; return; }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.replace(/^data: /, "").trim();
      if (!line) continue;
      let ev;
      try { ev = JSON.parse(line); } catch (e) { continue; }
      if (ev.type === "status") {
        status.textContent = ev.text;
      } else if (ev.type === "token") {
        status.textContent = "";
        answer += ev.text;
        body.innerHTML = md(answer);
        enhanceCode(body);
      } else if (ev.type === "citations") {
        if (ev.citations && ev.citations.length) {
          const src = document.createElement("div");
          src.className = "sources";
          src.innerHTML = "<b>Sources</b>" + ev.citations.map(
            (c) => `<a href="${c.url}" target="_blank">${c.title} @ ${Math.floor(c.start)}s</a>`
          ).join("");
          bot.appendChild(src);
        }
      } else if (ev.type === "error") {
        status.textContent = ev.text;
      }
      bot.scrollIntoView({ block: "end" });
    }
  }
}

function send() {
  const q = $("#q").value;
  $("#q").value = "";
  ask(q);
}
$("#send").onclick = send;
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.onclick = () => ask(chip.textContent);
});

loadChannels();
```

- [ ] **Step 5: Write `ytkb/web/static/styles.css`** (baseline dark theme — refine with frontend-design)

```css
:root {
  --bg: #0a0a0b; --panel: #161618; --border: #2a2a2e; --text: #ededef;
  --muted: #8a8a90; --bubble: #1f2937; --accent: #6366f1;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, sans-serif; min-height: 100vh;
}
#app { max-width: 820px; margin: 0 auto; min-height: 100vh; display: flex; flex-direction: column; padding: 0 1rem; }
#hero { margin-top: 22vh; text-align: center; }
#hero h1 { font-size: 2rem; font-weight: 700; }
#subtitle { color: var(--muted); }
body.chatting #hero { display: none; }
#messages { flex: 1; display: flex; flex-direction: column; gap: 1rem; padding: 1.5rem 0; overflow-y: auto; }
.msg.user { align-self: flex-end; background: var(--bubble); padding: .6rem 1rem; border-radius: 14px; max-width: 75%; }
.msg.assistant { align-self: flex-start; max-width: 100%; width: 100%; }
.msg.assistant .status { color: var(--muted); font-size: .9rem; min-height: 1rem; }
.msg.assistant .body { line-height: 1.6; }
.msg.assistant .body p { margin: .5rem 0; }
.sources { margin-top: .75rem; display: flex; flex-direction: column; gap: .25rem; font-size: .85rem; }
.sources b { color: var(--muted); font-weight: 600; }
.sources a { color: #34d399; text-decoration: none; }
pre { background: #0d1117; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
pre > code { display: block; padding: 1rem; overflow-x: auto; }
.code-header { display: flex; justify-content: space-between; align-items: center;
  padding: .4rem .8rem; background: #131820; color: var(--muted); font-size: .8rem; border-bottom: 1px solid var(--border); }
.code-header .copy { background: none; border: none; color: var(--muted); cursor: pointer; font-size: .8rem; }
#composer { position: sticky; bottom: 0; background: var(--bg); padding: 1rem 0 1.5rem; }
#bar { display: flex; align-items: center; gap: .5rem; background: var(--panel);
  border: 1px solid var(--border); border-radius: 16px; padding: .5rem .75rem; }
#channel-wrap { position: relative; display: flex; align-items: center; gap: .4rem; }
#channel-btn { width: 32px; height: 32px; border-radius: 10px; border: 1px solid var(--border);
  background: #232327; color: var(--text); font-size: 1.1rem; cursor: pointer; }
#channel-label { color: var(--muted); font-size: .85rem; }
#channel-menu { position: absolute; bottom: 40px; left: 0; background: var(--panel);
  border: 1px solid var(--border); border-radius: 10px; padding: .25rem; min-width: 160px; z-index: 10; }
#channel-menu button { display: block; width: 100%; text-align: left; background: none; border: none;
  color: var(--text); padding: .5rem .6rem; border-radius: 8px; cursor: pointer; }
#channel-menu button:hover { background: #232327; }
#q { flex: 1; background: none; border: none; color: var(--text); resize: none;
  font-family: inherit; font-size: 1rem; outline: none; max-height: 160px; }
#send { width: 32px; height: 32px; border-radius: 10px; border: none; background: var(--accent);
  color: white; cursor: pointer; font-size: 1rem; }
#chips { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .75rem; }
body.chatting #chips { display: none; }
.chip { background: var(--panel); border: 1px solid var(--border); color: var(--text);
  padding: .5rem .8rem; border-radius: 12px; cursor: pointer; font-size: .85rem; }
.chip:hover { border-color: var(--accent); }
```

- [ ] **Step 6: Run the served-page test + full suite**

Run: `uv run pytest tests/test_web.py -v && uv run pytest -q`
Expected: PASS (the served-page test + everything else).

- [ ] **Step 7: Manual verification**

Run `uv run kb web`, open http://127.0.0.1:8000, and confirm: dark empty state; `+` opens the channel menu and selecting sets the label; a chip click sends; live status appears then the answer streams in as Markdown; a code block (ask "show me an html button example") renders with a language label + Copy button; citations appear as clickable links.

- [ ] **Step 8: Commit**

```bash
git add ytkb/web/static/index.html ytkb/web/static/styles.css ytkb/web/static/app.js tests/test_web.py
git commit -m "feat: dark streaming chat UI with channel picker, markdown, code blocks"
```

---

### Task 5: Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README**

In `README.md`, replace the `kb web` usage line's comment to note the new UI:
```bash
uv run kb web                         # dark streaming chat UI at http://127.0.0.1:8000
```
And add a short line after the usage block:
```markdown
The web chat streams answers token-by-token, renders Markdown (with syntax-highlighted code
blocks), and lets you pick the channel from the **+** button. It loads `marked`/`DOMPurify`/
`highlight.js` from CDN, so it needs network access (as does the agent itself).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document the streaming web chat UI"
```

---

## Self-Review

**Spec coverage (spec section → task):**
- §2 decisions (CDN libs, SSE, stream-every-call, sanitized markdown, code-block highlight, split static files, `+` picker, CLI untouched) → Tasks 1–4 (CDN in 4, SSE in 3, mechanism in 1–2, sanitize+highlight in 4, picker/chips in 4, CLI untouched verified by leaving `answer()`/`/api/ask` intact).
- §3 visual (empty/conversation states, user pill, assistant block, status lines, code card, citations, channel picker) → Task 4 (`index.html` + `styles.css` + `app.js`).
- §4 protocol (status/token/citations/done/error) → Task 2 emits them; Task 3 serializes as SSE.
- §5.1 `stream_with_tools` → Task 1; §5.2 `answer_stream` → Task 2; §5.3 endpoint + StaticFiles → Task 3.
- §6 frontend structure → Task 4.
- §7 error handling (no key → error event; mid-stream error; client refuses empty/no-channel) → Task 3 (no-key), Task 2 (exception→error), Task 4 (`if (!question.trim() || !currentSlug) return`).
- §8 testing → Tasks 1–4 each TDD; manual verify in Task 4 Step 7.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `ToolCallStream`/`ToolCall`/`_Fn` (Task 1) are consumed by `answer_stream` (Task 2) exactly (`.tool_calls`, `tc.id`, `tc.function.name`, `tc.function.arguments`). Event dict shapes match across Tasks 2/3/4. `answer_stream(question, channel_title, store, llm, *, chat_model, top_k, max_steps=5)` signature is identical in Task 2 and its call site in Task 3. JS element ids in `index.html` (Task 4 Step 3) match those used in `app.js` (Step 4) and the test (Step 1): `#channel-btn`, `#channel-menu`, `#messages`, `#q`, `#send`, `.chip`, `[data-slug]`.
