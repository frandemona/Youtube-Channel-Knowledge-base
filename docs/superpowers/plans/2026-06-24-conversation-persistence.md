# Conversation Persistence + Multi-Chat UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist chats in a `conversations.db`, give the web UI a sidebar to start/resume/delete conversations, and make the agent multi-turn so follow-up questions use prior turns.

**Architecture:** A new `ytkb/conversations.py` (SQLite store) holds conversations + messages. `agent.answer_stream` gains an optional `history` and a sibling `generate_title`. `web/app.py` adds conversation endpoints and extends `/api/ask/stream` to lazily create a conversation, persist the user message, stream the answer while accumulating it, and persist the assistant message (+ generate a title) on `done`. The frontend adds a sidebar and conversation lifecycle.

**Tech Stack:** Existing `ytkb` stack — Python 3.12, stdlib `sqlite3`, FastAPI, the OpenRouter `LLMClient`. No new deps. Frontend uses the already-loaded CDN libs.

## Global Constraints

- Python **3.12**; run with `uv` (`uv run pytest`, `uv run kb web`).
- The non-streaming `agent.answer()`, `POST /api/ask`, and the CLI are **unchanged**. `answer_stream` stays backward-compatible (`history=None` ⇒ today's behavior).
- New `conversations.db` lives at `<data_dir>/conversations.db`, separate from per-channel `videos.db`. `connect` uses `check_same_thread=False` (the connection is used inside the streaming generator).
- A conversation's `slug` is **immutable**: follow-ups use the stored conversation's slug, never the request body's.
- New SSE events only **add** to the protocol: `conversation` (`id`) emitted first for a freshly-created chat; `title` (`text`) emitted after the first turn, before `done`.
- History is capped to the last **8 turns** (`MAX_TURNS = 8`), text-only.
- Tests must not hit the network/LLM: use temp DBs, scripted fake LLMs, and monkeypatched agent/conversations functions. TDD per task.

---

## File Structure

```
ytkb/conversations.py          # new: SQLite store (conversations + messages)
ytkb/agent.py                  # + history param on answer_stream; + generate_title
ytkb/web/app.py                # + conversation endpoints; extend /api/ask/stream (persist + title)
ytkb/web/static/index.html     # + sidebar markup
ytkb/web/static/styles.css     # + sidebar/layout styles
ytkb/web/static/app.js         # + conversation lifecycle (list/new/open/delete), send conversation_id
tests/test_conversations.py    # new
tests/test_agent_stream.py     # + history + generate_title tests
tests/test_web.py              # + conversation endpoint + persistence tests
```

---

### Task 1: `ytkb/conversations.py` — the store

**Files:**
- Create: `ytkb/conversations.py`
- Test: `tests/test_conversations.py`

**Interfaces:**
- Produces:
  - `connect(data_dir) -> sqlite3.Connection` (creates schema; `row_factory=Row`; `check_same_thread=False`).
  - `create_conversation(conn, slug: str) -> int`
  - `list_conversations(conn) -> list[dict]` — `{id, slug, title, updated_at}`, newest `updated_at` first.
  - `get_conversation(conn, id: int) -> dict | None` — `{id, slug, title, messages: [{role, content, citations}]}`.
  - `add_message(conn, conversation_id: int, role: str, content: str, citations: list | None = None) -> None` (bumps `updated_at`).
  - `set_title(conn, id: int, title: str) -> None`
  - `get_history(conn, conversation_id: int, max_turns: int) -> list[dict]` — last `max_turns*2` messages as `{role, content}`, oldest→newest.
  - `delete_conversation(conn, id: int) -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_conversations.py`:
```python
from ytkb import conversations as cv


def test_create_add_get_roundtrip(tmp_path):
    conn = cv.connect(tmp_path)
    cid = cv.create_conversation(conn, "ycombinator")
    assert isinstance(cid, int)
    cv.add_message(conn, cid, "user", "How do I find a co-founder?")
    cv.add_message(conn, cid, "assistant", "Find a technical one.",
                   citations=[{"video_id": "v1", "title": "T", "start": 1.0, "url": "u"}])
    conv = cv.get_conversation(conn, cid)
    assert conv["slug"] == "ycombinator"
    assert conv["title"] is None
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["citations"][0]["video_id"] == "v1"
    assert conv["messages"][0]["citations"] == []


def test_list_orders_by_updated_desc(tmp_path):
    conn = cv.connect(tmp_path)
    a = cv.create_conversation(conn, "c1")
    b = cv.create_conversation(conn, "c2")
    cv.add_message(conn, a, "user", "later")  # bumps a's updated_at after b
    ids = [c["id"] for c in cv.list_conversations(conn)]
    assert ids[0] == a
    assert set(ids) == {a, b}
    assert all("slug" in c and "title" in c for c in cv.list_conversations(conn))


def test_get_history_caps_and_strips_citations(tmp_path):
    conn = cv.connect(tmp_path)
    cid = cv.create_conversation(conn, "c")
    for i in range(5):
        cv.add_message(conn, cid, "user", f"q{i}")
        cv.add_message(conn, cid, "assistant", f"a{i}", citations=[{"x": 1}])
    hist = cv.get_history(conn, cid, max_turns=2)  # last 4 messages
    assert [m["content"] for m in hist] == ["q3", "a3", "q4", "a4"]
    assert all(set(m.keys()) == {"role", "content"} for m in hist)


def test_set_title_and_delete(tmp_path):
    conn = cv.connect(tmp_path)
    cid = cv.create_conversation(conn, "c")
    cv.add_message(conn, cid, "user", "hi")
    cv.set_title(conn, cid, "My chat")
    assert cv.get_conversation(conn, cid)["title"] == "My chat"
    cv.delete_conversation(conn, cid)
    assert cv.get_conversation(conn, cid) is None
    assert conn.execute("SELECT COUNT(*) c FROM messages WHERE conversation_id=?", (cid,)).fetchone()["c"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conversations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ytkb.conversations'`

- [ ] **Step 3: Implement `ytkb/conversations.py`**

```python
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(data_dir) -> sqlite3.Connection:
    path = Path(data_dir) / "conversations.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def create_conversation(conn, slug: str) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO conversations (slug, title, created_at, updated_at) VALUES (?,?,?,?)",
        (slug, None, now, now),
    )
    conn.commit()
    return cur.lastrowid


def list_conversations(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, slug, title, updated_at FROM conversations ORDER BY updated_at DESC, id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conn, id: int) -> dict | None:
    row = conn.execute("SELECT id, slug, title FROM conversations WHERE id=?", (id,)).fetchone()
    if row is None:
        return None
    msgs = conn.execute(
        "SELECT role, content, citations_json FROM messages WHERE conversation_id=? ORDER BY id",
        (id,),
    ).fetchall()
    return {
        "id": row["id"], "slug": row["slug"], "title": row["title"],
        "messages": [
            {"role": m["role"], "content": m["content"],
             "citations": json.loads(m["citations_json"]) if m["citations_json"] else []}
            for m in msgs
        ],
    }


def add_message(conn, conversation_id: int, role: str, content: str, citations: list | None = None) -> None:
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, citations_json, created_at) VALUES (?,?,?,?,?)",
        (conversation_id, role, content, json.dumps(citations) if citations else None, _now()),
    )
    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conversation_id))
    conn.commit()


def set_title(conn, id: int, title: str) -> None:
    conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, id))
    conn.commit()


def get_history(conn, conversation_id: int, max_turns: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
        (conversation_id, max_turns * 2),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def delete_conversation(conn, id: int) -> None:
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (id,))
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_conversations.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add ytkb/conversations.py tests/test_conversations.py
git commit -m "feat: conversations store (sqlite conversations + messages)"
```

---

### Task 2: Agent — `history` param + `generate_title`

**Files:**
- Modify: `ytkb/agent.py`
- Test: `tests/test_agent_stream.py` (add tests; keep existing)

**Interfaces:**
- Consumes: `LLMClient.stream_with_tools`, `LLMClient.complete` (existing).
- Produces:
  - `answer_stream(question, channel_title, store, llm, *, chat_model, top_k, history=None, max_steps=5)` — now seeds prior `history` turns. `history` is `list[{"role","content"}] | None`.
  - `generate_title(llm, model: str, question: str, answer: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_stream.py`:
```python
def test_answer_stream_includes_history_in_messages():
    from types import SimpleNamespace
    from ytkb.agent import answer_stream

    class _S:
        def __init__(self, tokens):
            self._t = tokens
            self.tool_calls = []
        def __iter__(self):
            return iter(self._t)

    captured = {}

    class CapturingLLM:
        def stream_with_tools(self, messages, model, tools):
            captured["messages"] = messages
            return _S(["ok"])

    history = [{"role": "user", "content": "prev q"}, {"role": "assistant", "content": "prev a"}]
    list(answer_stream("new q", "C", FakeStore(), CapturingLLM(),
                       chat_model="m", top_k=5, history=history))
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured["messages"][-1]["content"] == "new q"
    assert captured["messages"][1]["content"] == "prev q"


def test_generate_title_uses_llm_then_falls_back():
    from ytkb.agent import generate_title

    class TitleLLM:
        def complete(self, messages, model):
            return '"Finding a co-founder"\n'

    assert generate_title(TitleLLM(), "m", "How do I find a co-founder?", "Find one.") == "Finding a co-founder"

    class EmptyLLM:
        def complete(self, messages, model):
            return "   "

    # empty LLM output -> fall back to first words of the question
    assert generate_title(EmptyLLM(), "m", "How do I find a co-founder today please", "x") == "How do I find a co-founder"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_stream.py -k "history or title" -v`
Expected: FAIL (`history` kwarg unexpected / `generate_title` not defined)

- [ ] **Step 3: Implement in `ytkb/agent.py`**

Change the `answer_stream` signature and the message seeding (only these two lines change inside `answer_stream`):
```python
def answer_stream(question, channel_title, store, llm, *, chat_model, top_k, history=None, max_steps=5):
    try:
        specs, dispatch = build_tools(store, top_k)
        messages = [{"role": "system", "content": SYSTEM.format(channel=channel_title)}]
        messages += history or []
        messages.append({"role": "user", "content": question})
        cited: dict[str, Citation] = {}
        # ... rest of the function body is UNCHANGED ...
```
Then add `generate_title` at the end of the module:
```python
TITLE_PROMPT = (
    "Write a short 3-6 word title (no quotes, no trailing punctuation) summarizing this "
    "conversation.\n\nUser: {q}\nAssistant: {a}\n\nTitle:"
)


def generate_title(llm, model: str, question: str, answer: str) -> str:
    title = ""
    try:
        raw = llm.complete(
            [{"role": "user", "content": TITLE_PROMPT.format(q=question, a=answer[:500])}], model
        )
        title = (raw or "").strip().strip('"').splitlines()[0].strip()[:60] if raw and raw.strip() else ""
    except Exception:
        title = ""
    if not title:
        title = " ".join(question.split()[:6])
    return title
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_stream.py -v`
Expected: PASS (existing stream tests + 2 new)

- [ ] **Step 5: Commit**

```bash
git add ytkb/agent.py tests/test_agent_stream.py
git commit -m "feat: answer_stream history param + generate_title"
```

---

### Task 3: Web API — conversation endpoints + persisting stream

**Files:**
- Modify: `ytkb/web/app.py`
- Test: `tests/test_web.py` (add tests; keep existing)

**Interfaces:**
- Consumes: `ytkb.conversations` (Task 1), `agent.answer_stream`/`generate_title` (Task 2), `sync`.
- Produces:
  - `GET /api/conversations` → `{"conversations": [...]}`.
  - `GET /api/conversations/{id}` → the conversation dict (404 if missing).
  - `DELETE /api/conversations/{id}` → `{"ok": true}`.
  - `POST /api/ask/stream` extended: body `{slug, question, conversation_id: int | null}`; lazily creates, persists, emits `conversation`/`title` events.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web.py`:
```python
def test_conversations_crud(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    from ytkb import conversations as cv
    conn = cv.connect(tmp_path)
    cid = cv.create_conversation(conn, "yc")
    cv.add_message(conn, cid, "user", "hi")
    client = TestClient(create_app())
    listed = client.get("/api/conversations").json()["conversations"]
    assert listed[0]["id"] == cid and listed[0]["slug"] == "yc"
    got = client.get(f"/api/conversations/{cid}").json()
    assert got["messages"][0]["content"] == "hi"
    assert client.get("/api/conversations/99999").status_code == 404
    assert client.delete(f"/api/conversations/{cid}").json() == {"ok": True}
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_ask_stream_creates_persists_and_titles(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))

    class FakeCtx:
        store = object()
        llm = object()

    from ytkb.channel import ChannelInfo, ChannelFilters
    info = ChannelInfo("UC1", "@yc", "Y Combinator", "u")
    monkeypatch.setattr(webapp.sync, "build_context", lambda cfg, slug: FakeCtx())
    monkeypatch.setattr(webapp.sync, "load_channel", lambda cfg, slug: (info, ChannelFilters()))
    monkeypatch.setattr(webapp.agent, "answer_stream", lambda *a, **k: iter([
        {"type": "token", "text": "Hello"},
        {"type": "citations", "citations": []},
        {"type": "done"},
    ]))
    monkeypatch.setattr(webapp.agent, "generate_title", lambda *a, **k: "A Title")

    client = TestClient(create_app())
    body = client.post("/api/ask/stream", json={"slug": "yc", "question": "hi", "conversation_id": None}).text
    assert '"type": "conversation"' in body
    assert '"type": "title"' in body and "A Title" in body
    assert '"type": "done"' in body

    # the conversation + both messages were persisted
    from ytkb import conversations as cv
    conn = cv.connect(tmp_path)
    convs = cv.list_conversations(conn)
    assert len(convs) == 1 and convs[0]["title"] == "A Title"
    got = cv.get_conversation(conn, convs[0]["id"])
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]
    assert got["messages"][1]["content"] == "Hello"
```
(`tests/test_web.py` already imports `from fastapi.testclient import TestClient`, `from ytkb.web.app import create_app`, `import ytkb.web.app as webapp`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web.py::test_conversations_crud -v`
Expected: FAIL (routes missing)

- [ ] **Step 3: Implement in `ytkb/web/app.py`**

Add the import and `MAX_TURNS`, extend `AskReq`, and add the endpoints. Replace the existing `ask_stream` with the persisting version. Full updated file additions:
```python
from fastapi import FastAPI, HTTPException

from .. import sync, agent
from ..config import load_config
from .. import conversations as cv

MAX_TURNS = 8


class AskReq(BaseModel):
    slug: str
    question: str
    conversation_id: int | None = None
```
Inside `create_app`, add these routes (keep `/`, `/api/channels`, `/api/ask`, `/static`):
```python
    @app.get("/api/conversations")
    def conversations_list():
        conn = cv.connect((cfg or load_config()).data_dir)
        return {"conversations": cv.list_conversations(conn)}

    @app.get("/api/conversations/{cid}")
    def conversation_get(cid: int):
        conn = cv.connect((cfg or load_config()).data_dir)
        conv = cv.get_conversation(conn, cid)
        if conv is None:
            raise HTTPException(status_code=404, detail="not found")
        return conv

    @app.delete("/api/conversations/{cid}")
    def conversation_delete(cid: int):
        conn = cv.connect((cfg or load_config()).data_dir)
        cv.delete_conversation(conn, cid)
        return {"ok": True}
```
And replace `ask_stream` with:
```python
    @app.post("/api/ask/stream")
    def ask_stream(req: AskReq):
        c = cfg or load_config()
        conn = cv.connect(c.data_dir)

        # Resolve the effective conversation + immutable slug.
        existing = cv.get_conversation(conn, req.conversation_id) if req.conversation_id else None
        slug = existing["slug"] if existing else req.slug
        had_title = bool(existing and existing["title"])

        info, _ = sync.load_channel(c, slug)
        ctx = sync.build_context(c, slug)
        if ctx.llm is None:
            return StreamingResponse(_sse(iter([
                {"type": "error", "text": "No OPENROUTER_API_KEY configured in data/.env"},
                {"type": "done"},
            ])), media_type="text/event-stream")

        is_new = existing is None
        conv_id = existing["id"] if existing else cv.create_conversation(conn, slug)
        history = cv.get_history(conn, conv_id, MAX_TURNS) if existing else []
        cv.add_message(conn, conv_id, "user", req.question)

        def gen():
            if is_new:
                yield {"type": "conversation", "id": conv_id}
            answer_text, citations, errored = "", [], False
            for ev in agent.answer_stream(req.question, info.title, ctx.store, ctx.llm,
                                          chat_model=c.chat_model, top_k=c.top_k, history=history):
                t = ev["type"]
                if t == "token":
                    answer_text += ev["text"]
                elif t == "citations":
                    citations = ev["citations"]
                elif t == "error":
                    errored = True
                if t == "done":
                    if not errored:
                        cv.add_message(conn, conv_id, "assistant", answer_text, citations)
                        if not had_title:
                            title = agent.generate_title(ctx.llm, c.chat_model, req.question, answer_text)
                            cv.set_title(conn, conv_id, title)
                            yield {"type": "title", "text": title}
                    yield ev
                else:
                    yield ev

        return StreamingResponse(_sse(gen()), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web.py -v`
Expected: PASS (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add ytkb/web/app.py tests/test_web.py
git commit -m "feat: conversation endpoints + persisting streaming ask"
```

---

### Task 4: Frontend — sidebar + conversation lifecycle

**Files:**
- Modify: `ytkb/web/static/index.html`, `ytkb/web/static/styles.css`, `ytkb/web/static/app.js`
- Test: add one served-page assertion to `tests/test_web.py`; the rest is manual verification.

**Interfaces:**
- Consumes: `GET/DELETE /api/conversations`, `GET /api/conversations/{id}`, `POST /api/ask/stream` (with `conversation_id`, new `conversation`/`title` events).
- Produces: the sidebar UI + conversation lifecycle.

> **Implementer note:** implement the `app.js` behavior as written (the ids/classes other code and the test depend on: `#sidebar`, `#new-chat`, `#conversations`, plus the existing `#channel-btn`, `#channel-menu`, `#messages`, `#hero`, `#q`, `#send`, `.chip`). Use the **frontend-design** skill to style the sidebar cohesively with the existing dark theme (do not change ids/classes or behavior).

- [ ] **Step 1: Write the failing served-page test**

Add to `tests/test_web.py`:
```python
def test_index_has_sidebar(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())
    html = client.get("/").text
    assert 'id="sidebar"' in html and 'id="new-chat"' in html and 'id="conversations"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web.py::test_index_has_sidebar -v`
Expected: FAIL (no sidebar markup yet)

- [ ] **Step 3: Update `ytkb/web/static/index.html`**

Wrap the existing `<main id="app">…</main>` in a flex shell with the sidebar before it. The new `<body>` contents:
```html
  <div id="shell">
    <aside id="sidebar">
      <button id="new-chat" type="button">+ New chat</button>
      <div id="conversations"></div>
    </aside>
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
  </div>
```
Keep the existing `<head>` (CDN libs + styles.css) and the `<script src="/static/app.js">` at the end of `<body>`.

- [ ] **Step 4: Rewrite `ytkb/web/static/app.js`** (full file — adds conversation lifecycle)

```javascript
const $ = (s) => document.querySelector(s);
let currentSlug = null;
let conversationId = null;

function md(text) { return DOMPurify.sanitize(marked.parse(text || "")); }

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

function citationsHtml(citations) {
  if (!citations || !citations.length) return "";
  return DOMPurify.sanitize('<div class="sources"><b>Sources</b>' + citations.map(
    (c) => `<a href="${c.url}" target="_blank" rel="noopener noreferrer">${c.title} @ ${Math.floor(c.start)}s</a>`
  ).join("") + "</div>");
}

async function loadChannels() {
  const d = await (await fetch("/api/channels")).json();
  const menu = $("#channel-menu");
  menu.innerHTML = (d.channels || []).map((c) => `<button data-slug="${c}">${c}</button>`).join("");
  menu.querySelectorAll("[data-slug]").forEach((b) => {
    b.onclick = () => { setChannel(b.dataset.slug); menu.hidden = true; };
  });
  if (d.channels && d.channels.length && !currentSlug) setChannel(d.channels[0]);
}

function setChannel(slug) {
  currentSlug = slug;
  $("#channel-label").textContent = slug || "";
}

function lockChannel(locked) {
  $("#channel-btn").disabled = locked;
  $("#channel-btn").style.opacity = locked ? "0.4" : "1";
  if (locked) $("#channel-menu").hidden = true;
}

$("#channel-btn").onclick = () => {
  if ($("#channel-btn").disabled) return;
  $("#channel-menu").hidden = !$("#channel-menu").hidden;
};

async function loadConversations() {
  const d = await (await fetch("/api/conversations")).json();
  const list = $("#conversations");
  list.innerHTML = "";
  (d.conversations || []).forEach((c) => {
    const row = document.createElement("div");
    row.className = "conv" + (c.id === conversationId ? " active" : "");
    row.innerHTML = `<span class="conv-title">${DOMPurify.sanitize(c.title || "New chat")}</span>` +
                    `<span class="conv-badge">${DOMPurify.sanitize(c.slug)}</span>` +
                    `<button class="conv-del" title="Delete">×</button>`;
    row.querySelector(".conv-title").onclick = () => openConversation(c.id);
    row.querySelector(".conv-badge").onclick = () => openConversation(c.id);
    row.querySelector(".conv-del").onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (c.id === conversationId) newChat();
      loadConversations();
    };
    list.appendChild(row);
  });
}

function newChat() {
  conversationId = null;
  $("#messages").innerHTML = "";
  document.body.classList.remove("chatting");
  lockChannel(false);
  loadConversations();
}
$("#new-chat").onclick = newChat;

function addMessage(role) {
  document.body.classList.add("chatting");
  const el = document.createElement("div");
  el.className = "msg " + role;
  $("#messages").appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

function renderAssistant(content, citations) {
  const bot = addMessage("assistant");
  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = md(content);
  bot.appendChild(body);
  enhanceCode(body);
  if (citations && citations.length) {
    const src = document.createElement("div");
    src.innerHTML = citationsHtml(citations);
    bot.appendChild(src.firstChild);
  }
}

async function openConversation(id) {
  const conv = await (await fetch(`/api/conversations/${id}`)).json();
  conversationId = id;
  setChannel(conv.slug);
  lockChannel(true);
  $("#messages").innerHTML = "";
  document.body.classList.add("chatting");
  conv.messages.forEach((m) => {
    if (m.role === "user") addMessage("user").textContent = m.content;
    else renderAssistant(m.content, m.citations);
  });
  loadConversations();
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
      body: JSON.stringify({ slug: currentSlug, question, conversation_id: conversationId }),
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
      if (ev.type === "conversation") {
        conversationId = ev.id;
        lockChannel(true);
        loadConversations();
      } else if (ev.type === "status") {
        status.textContent = ev.text;
      } else if (ev.type === "token") {
        status.textContent = "";
        answer += ev.text;
        body.innerHTML = md(answer);
      } else if (ev.type === "citations") {
        if (ev.citations && ev.citations.length) {
          const src = document.createElement("div");
          src.innerHTML = citationsHtml(ev.citations);
          bot.appendChild(src.firstChild);
        }
      } else if (ev.type === "title") {
        loadConversations();
      } else if (ev.type === "error") {
        status.textContent = ev.text;
      }
      bot.scrollIntoView({ block: "end" });
    }
  }
  body.innerHTML = md(answer);
  enhanceCode(body);
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
document.querySelectorAll(".chip").forEach((chip) => { chip.onclick = () => ask(chip.textContent); });

loadChannels();
loadConversations();
```

- [ ] **Step 5: Add sidebar styles to `ytkb/web/static/styles.css`** (baseline — refine with frontend-design)

```css
#shell { display: flex; min-height: 100vh; }
#sidebar { width: 240px; flex-shrink: 0; background: #111113; border-right: 1px solid var(--border);
  padding: 1rem .75rem; display: flex; flex-direction: column; gap: .5rem; overflow-y: auto; }
#new-chat { background: var(--panel); border: 1px solid var(--border); color: var(--text);
  padding: .6rem; border-radius: 10px; cursor: pointer; font-size: .9rem; }
#new-chat:hover { border-color: var(--accent); }
#conversations { display: flex; flex-direction: column; gap: .15rem; margin-top: .5rem; }
.conv { display: flex; align-items: center; gap: .4rem; padding: .5rem .55rem; border-radius: 8px;
  cursor: pointer; font-size: .85rem; }
.conv:hover { background: #1c1c20; }
.conv.active { background: #232327; }
.conv-title { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-badge { font-size: .65rem; color: var(--muted); background: #232327; padding: .1rem .35rem;
  border-radius: 6px; }
.conv-del { background: none; border: none; color: var(--muted); cursor: pointer; opacity: 0; }
.conv:hover .conv-del { opacity: 1; }
#app { flex: 1; }
#channel-btn:disabled { cursor: default; }
```
(The existing `#app { max-width: 820px; margin: 0 auto; … }` rule should keep centering the chat column within the now-flexed main area; if the implementer's frontend-design polish adjusts the layout, preserve the centered chat column and a fixed-width sidebar.)

- [ ] **Step 6: Run the served-page test + full suite**

Run: `uv run pytest tests/test_web.py -v && uv run pytest -q`
Expected: PASS.

- [ ] **Step 7: Manual verification**

`uv run kb web` → confirm: sidebar with "New chat"; ask a question → a row appears and gets an LLM title; reload → conversation persists and resumes; open another conversation → channel switches and the `+` picker locks; "New chat" unlocks the picker; delete removes a row.

- [ ] **Step 8: Commit**

```bash
git add ytkb/web/static/index.html ytkb/web/static/app.js ytkb/web/static/styles.css tests/test_web.py
git commit -m "feat: conversation sidebar (new/open/delete) with persisted history"
```

---

### Task 5: Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README**

Add a short paragraph after the existing web-UI note in `README.md`:
```markdown
Conversations are saved to `data/conversations.db`: the sidebar lists every chat (across all
channels), **New chat** starts a fresh one, and opening a past chat resumes it (follow-up
questions use the prior turns). A chat's channel is fixed when it's created.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document conversation persistence and the sidebar"
```

---

## Self-Review

**Spec coverage (spec section → task):**
- §3 storage (`conversations.py`, schema, all functions) → Task 1.
- §4 agent `history` + `generate_title` → Task 2.
- §5 API (conversations list/get/delete, extended `/api/ask/stream` with no-key short-circuit, lazy create, persist user→stream→persist assistant, title on first turn, immutable slug) → Task 3.
- §6 frontend (sidebar, new/open/delete, send conversation_id, lock picker on open, `conversation`/`title` handling) → Task 4.
- §7 error handling (no key → error+done, persist nothing; unknown conversation_id → treated as new via `existing is None`; 404 on missing GET; title fallback) → Tasks 1–3 (`get_conversation` returns None → `is_new` path; `generate_title` fallback in Task 2).
- §8 testing → each task is TDD; manual verify in Task 4.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `conversations.*` signatures (Task 1) are used identically in Task 3 (`connect`, `create_conversation`, `get_conversation` returning `{id,slug,title,messages}`, `add_message(conn, id, role, content, citations)`, `get_history(conn, id, MAX_TURNS)`, `set_title`, `delete_conversation`). `answer_stream(..., history=...)` and `generate_title(llm, model, question, answer)` (Task 2) match their call sites in Task 3. SSE event shapes (`conversation`/`title` + existing) are produced in Task 3 and consumed in Task 4's `app.js`. Frontend element ids (`#sidebar`, `#new-chat`, `#conversations`, `#channel-btn`, `#messages`, etc.) match across index.html (Task 4 Step 3), app.js (Step 4), and the test (Step 1).
