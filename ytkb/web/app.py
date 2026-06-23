from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .. import sync, agent
from ..config import load_config

STATIC = Path(__file__).parent / "static"


class AskReq(BaseModel):
    slug: str
    question: str


def create_app(cfg=None) -> FastAPI:
    app = FastAPI(title="ytkb")

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

    return app
