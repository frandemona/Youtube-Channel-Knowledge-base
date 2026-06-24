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


def test_index_has_streaming_ui(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())
    html = client.get("/").text
    assert 'id="messages"' in html and 'id="channel-btn"' in html
    assert "/static/app.js" in html
    # static asset is served
    assert client.get("/static/app.js").status_code == 200


def test_index_has_sidebar(monkeypatch, tmp_path):
    monkeypatch.setenv("YTKB_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())
    html = client.get("/").text
    assert 'id="sidebar"' in html and 'id="new-chat"' in html and 'id="conversations"' in html


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
