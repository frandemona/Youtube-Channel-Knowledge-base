# Conversation Persistence + Multi-Chat Web UI — Design Spec

**Date:** 2026-06-24
**Status:** Approved (pending final spec review)
**Builds on:** [2026-06-23-web-chat-redesign-design.md](2026-06-23-web-chat-redesign-design.md)

## 1. Purpose

Today the web chat is stateless: `agent.answer_stream` answers one question with no memory, and
the UI keeps the conversation only in the DOM (lost on reload). Add persistent, multi-turn
conversations: store chats, list them in a sidebar, start new ones, resume old ones, and let
follow-up questions use prior turns as context.

Non-goals (YAGNI): auth/multi-user, conversation search, summarizing long histories, sharing/
export, editing past messages, renaming titles by hand (auto-title only), CLI conversation
support (the CLI stays single-shot).

## 2. Key decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Title | **LLM-generated** — one small `complete()` call after the first turn |
| Sidebar scope | **All channels combined**, each row labeled with its channel |
| Channel per conversation | **Immutable** — set at creation; the `+` picker is interactive only for a new chat; an open conversation shows its channel read-only |
| Follow-up context | **Re-search fresh each turn**; history is prior user/assistant **text** turns only |
| History budget | **Cap to the last N turns** (default 8); no summarization |
| Storage | A single **`data/conversations.db`** (separate from per-channel `videos.db`); each conversation carries its `slug` |
| Creation | **Lazy** — "New chat" creates nothing server-side; the first send creates the conversation and returns its id via a `conversation` SSE event |
| Existing protocol | Unchanged; we only **add** `conversation` and `title` SSE events |

## 3. Storage — `ytkb/conversations.py`

SQLite at `data/conversations.db`:
```sql
CREATE TABLE conversations (
  id INTEGER PRIMARY KEY,
  slug TEXT NOT NULL,
  title TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL,
  role TEXT NOT NULL,             -- 'user' | 'assistant'
  content TEXT NOT NULL,
  citations_json TEXT,           -- JSON list[dict] for assistant turns; NULL for user
  created_at TEXT NOT NULL
);
```

Functions (each takes a `data_dir`/`Config` and opens its own connection, mirroring `db.py`):
- `connect(data_dir) -> Connection` (creates schema).
- `create_conversation(conn, slug) -> int` (returns new id; `title` NULL initially).
- `list_conversations(conn) -> list[dict]` — `{id, slug, title, updated_at}`, newest `updated_at` first.
- `get_conversation(conn, id) -> dict | None` — `{id, slug, title, messages: [{role, content, citations}]}`.
- `add_message(conn, conversation_id, role, content, citations=None) -> None` (also bumps the conversation's `updated_at`).
- `set_title(conn, id, title) -> None`.
- `get_history(conn, conversation_id, max_turns) -> list[dict]` — the last `max_turns*2` messages as `[{role, content}]` (text only, no citations), oldest→newest.
- `delete_conversation(conn, id) -> None` (deletes the conversation and its messages).

A conversation's `slug` is never updated after creation (immutability is enforced by simply having
no update path for it).

## 4. Agent — multi-turn

`answer_stream` gains an optional keyword `history`:
`answer_stream(question, channel_title, store, llm, *, chat_model, top_k, history=None, max_steps=5)`.
It seeds `messages = [system] + (history or []) + [{"role":"user","content":question}]`. `history`
is a list of `{role, content}` text turns. Everything else (tool loop, status/token/citations/done,
error wrapper, max-steps fallback) is unchanged. With `history=None` behavior is identical to today,
so the existing tests and call sites keep working.

Title generation lives in the agent module too:
`generate_title(llm, model, question, answer) -> str` — one `llm.complete()` call with a prompt
that asks for a 3–6 word title; returns a trimmed single line (fallback: first ~6 words of the
question if the call returns empty).

## 5. API — `ytkb/web/app.py`

A per-app conversations connection is opened from `cfg.data_dir` (lazily, like `load_config`).

- `GET /api/conversations` → `{"conversations": list_conversations(...)}`.
- `GET /api/conversations/{id}` → the conversation dict, or 404 if missing.
- `DELETE /api/conversations/{id}` → `{"ok": true}`.
- `POST /api/ask/stream` — body extended to `{slug, question, conversation_id: int | null}`:
  0. Build context for the effective slug; if `ctx.llm is None`, emit `error`+`done` and return
     **without creating a conversation or persisting anything**.
  1. If `conversation_id` is null → `create_conversation(slug)`; the SSE stream's first event is
     `{"type":"conversation","id":<id>}`. The `slug` used is the request's slug (new chats only).
  2. If `conversation_id` is given → use the **stored conversation's slug** (ignore any slug in the
     body — channel is immutable), load `get_history(conv_id, N)`.
  3. `add_message(conv_id, "user", question)`.
  4. Stream `answer_stream(question, channel_title, store, llm, history=..., ...)`, wrapping it so the
     server **accumulates** answer text (from `token` events) and citations (from the `citations`
     event) while passing every event through to the client.
  5. On the agent's `done`: `add_message(conv_id, "assistant", answer_text, citations)`. If the
     conversation has no title yet (first turn), call `generate_title(...)`, `set_title(...)`, and
     emit `{"type":"title","text":<title>}` **before** forwarding `done`.
  - On `error`: forward it; do not persist an assistant message (the user message stays, so a retry
    re-asks cleanly).

The build-context (`sync.build_context`) is keyed on the conversation's slug, so retrieval always
hits the right channel's index.

## 6. Frontend — `ytkb/web/static/`

- **Sidebar** (left, collapsible): a **"New chat"** button and the conversation list from
  `GET /api/conversations`; each row shows the title (or "New chat" until titled) and a small
  channel badge, with a delete affordance on hover.
- **New chat**: clears the message area, sets `conversationId = null`, and **enables** the `+`
  channel picker (the user chooses the channel for this new chat).
- **Open a conversation** (sidebar click): `GET /api/conversations/{id}`, render its messages
  (Markdown + citations), set `conversationId` and the active channel to the conversation's slug,
  and **lock** the `+` picker (channel shown read-only).
- **Ask**: `POST /api/ask/stream` with `{slug, question, conversation_id}`. Handle the new events:
  on `conversation` set `conversationId` and refresh the sidebar (the new untitled row appears);
  on `title` update that row's label. Server persists, so a reload restores everything.
- **Delete**: calls `DELETE`, removes the row, and clears the view if it was active.
- On load: fetch the sidebar; show the empty hero state with the picker enabled (no conversation
  open) until the user starts or opens one.

## 7. Error handling
- No API key (`ctx.llm is None`): the stream emits an `error` event (as today); no messages persisted
  beyond the user turn — actually, persist nothing in this case (skip the `add_message` user write
  when `llm is None`, emitting only `error`+`done`).
- Missing/unknown `conversation_id` on ask: treat as a new conversation (create one) rather than 500.
- `GET /api/conversations/{id}` for a missing id → 404.
- Title generation failure: fall back to the first-question snippet; never block the answer.

## 8. Testing
- **`conversations.py`**: create → add user+assistant messages (with citations) → `get_conversation`
  returns them in order; `list_conversations` orders by `updated_at` desc and includes slug/title;
  `get_history` caps to the last N turns and strips citations; `set_title`; `delete_conversation`
  removes conversation + messages.
- **`agent.answer_stream` with history**: a scripted fake LLM seeded with `history` still yields
  status/token/citations/done; assert the history is included in the messages passed to the LLM
  (capture via the fake). **`generate_title`**: a fake `complete()` returns a title → asserted
  trimmed; empty → falls back to the question snippet.
- **Web**: `TestClient` over `GET/DELETE /api/conversations`, `GET /api/conversations/{id}` (+404);
  `POST /api/ask/stream` with `conversation_id: null` (monkeypatched `answer_stream`) asserts a
  `conversation` event then a `title` event in the SSE body and that a user+assistant message were
  persisted; a second call with the returned id asserts history is loaded and passed through. Existing
  web/agent tests stay green.
- **Frontend**: manual verification — new chat, ask, see it titled in the sidebar, reload and resume,
  open another conversation (channel switches + picker locks), delete.

## 9. Out of scope / follow-ups
Conversation search, manual rename, export/share, history summarization for very long chats, and the
previously-deferred items (`last_sync`, run-error persistence, the minor frontend polish).
