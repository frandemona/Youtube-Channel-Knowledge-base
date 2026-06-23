from fastapi.testclient import TestClient
from ytkb.web.app import create_app
from ytkb.channel import ChannelInfo, ChannelFilters
from ytkb.models import Answer, Citation
from ytkb import sync, agent
import ytkb.web.app as webapp


def test_channels_and_ask(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(webapp.sync, "list_channels", lambda cfg: ["y-combinator"])

    class FakeCtx:
        store = object()
        llm = object()

    info = ChannelInfo("UC1", "@yc", "Y Combinator", "u")
    monkeypatch.setattr(webapp.sync, "build_context", lambda cfg, slug: FakeCtx())
    monkeypatch.setattr(webapp.sync, "load_channel", lambda cfg, slug: (info, ChannelFilters()))
    monkeypatch.setattr(webapp.agent, "answer", lambda *a, **k: Answer(
        "Find a cofounder.", [Citation("v1", "Cofounders", 42.0, "https://youtu.be/v1?t=42")]))

    client = TestClient(create_app())
    assert client.get("/api/channels").json() == {"channels": ["y-combinator"]}

    r = client.post("/api/ask", json={"slug": "y-combinator", "question": "how?"})
    body = r.json()
    assert body["answer"] == "Find a cofounder."
    assert body["citations"][0]["url"] == "https://youtu.be/v1?t=42"


def test_index_served(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
