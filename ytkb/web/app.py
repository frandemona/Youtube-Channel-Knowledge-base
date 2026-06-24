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
