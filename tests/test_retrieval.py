import json
from ytkb.retrieval import build_tools
from ytkb.models import ChunkHit


class FakeStore:
    def keyword_search(self, query, k):
        return [ChunkHit("v1", "T", 12.0, "cofounder advice", 1.0)]

    def semantic_search(self, query, k):
        return [ChunkHit("v2", "T2", 30.0, "co-founder concept", 0.1)]

    def read_around(self, video_id, around_ts, window=90.0):
        return "full context here"

    def list_videos(self, contains=None):
        return [{"video_id": "v1", "title": "T", "upload_date": "20240101"}]


def test_tool_specs_cover_all_tools():
    specs, _ = build_tools(FakeStore(), top_k=5)
    names = {s["function"]["name"] for s in specs}
    assert names == {"keyword_search", "semantic_search", "read_transcript", "list_videos"}


def test_dispatch_keyword_search_returns_json_with_citation_fields():
    _, dispatch = build_tools(FakeStore(), top_k=5)
    out = json.loads(dispatch("keyword_search", {"query": "cofounder"}))
    assert out[0]["video_id"] == "v1"
    assert out[0]["start"] == 12.0
    assert "text" in out[0]


def test_dispatch_read_transcript():
    _, dispatch = build_tools(FakeStore(), top_k=5)
    out = dispatch("read_transcript", {"video_id": "v1", "around_ts": 12.0})
    assert "full context" in out


def test_read_transcript_missing_video_id_returns_error():
    _, dispatch = build_tools(FakeStore(), top_k=5)
    out = json.loads(dispatch("read_transcript", {}))
    assert "error" in out
