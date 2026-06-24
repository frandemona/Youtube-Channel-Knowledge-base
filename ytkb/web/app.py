import json
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import sync, agent
from ..config import load_config
from .. import conversations as cv

STATIC = Path(__file__).parent / "static"

MAX_TURNS = 8


class AskReq(BaseModel):
    slug: str
    question: str
    conversation_id: int | None = None


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

    @app.get("/api/conversations")
    def conversations_list():
        conn = cv.connect((cfg or load_config()).data_dir)
        try:
            return {"conversations": cv.list_conversations(conn)}
        finally:
            conn.close()

    @app.get("/api/conversations/{cid}")
    def conversation_get(cid: int):
        conn = cv.connect((cfg or load_config()).data_dir)
        try:
            conv = cv.get_conversation(conn, cid)
            if conv is None:
                raise HTTPException(status_code=404, detail="not found")
            return conv
        finally:
            conn.close()

    @app.delete("/api/conversations/{cid}")
    def conversation_delete(cid: int):
        conn = cv.connect((cfg or load_config()).data_dir)
        try:
            cv.delete_conversation(conn, cid)
            return {"ok": True}
        finally:
            conn.close()

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
            conn.close()
            return StreamingResponse(_sse(iter([
                {"type": "error", "text": "No OPENROUTER_API_KEY configured in data/.env"},
                {"type": "done"},
            ])), media_type="text/event-stream")

        is_new = existing is None
        conv_id = existing["id"] if existing else cv.create_conversation(conn, slug)
        history = cv.get_history(conn, conv_id, MAX_TURNS) if existing else []
        cv.add_message(conn, conv_id, "user", req.question)

        def gen():
            try:
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
            finally:
                conn.close()

        return StreamingResponse(_sse(gen()), media_type="text/event-stream")

    return app
