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


def test_answer_stream_forces_final_answer_when_max_steps_exhausted():
    class AlwaysToolsLLM:
        def stream_with_tools(self, messages, model, tools):
            if not tools:  # the forced final tool-less call
                return _Stream(["Final answer."], [])
            tc = SimpleNamespace(id="c", function=SimpleNamespace(name="keyword_search", arguments='{"query":"x"}'))
            return _Stream([], [tc])

    events = list(answer_stream("q", "C", FakeStore(), AlwaysToolsLLM(), chat_model="m", top_k=5, max_steps=2))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == "Final answer."
    assert events[-1]["type"] == "done"
