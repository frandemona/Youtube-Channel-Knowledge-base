from types import SimpleNamespace
from ytkb.agent import answer
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


class ScriptedLLM:
    """First call -> request keyword_search; second call -> final answer."""
    def __init__(self):
        self.n = 0

    def chat_with_tools(self, messages, model, tools):
        self.n += 1
        if self.n == 1:
            tc = SimpleNamespace(
                id="call1",
                function=SimpleNamespace(name="keyword_search", arguments='{"query": "cofounder"}'),
            )
            return SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(content="You should find a technical cofounder.", tool_calls=None)


def test_agent_runs_tool_then_answers_with_citation():
    ans = answer("How do I find a cofounder?", "Y Combinator", FakeStore(), ScriptedLLM(),
                 chat_model="m", top_k=5)
    assert "technical cofounder" in ans.text
    assert ans.citations[0].url == "https://youtu.be/vid123?t=42"


def test_agent_stops_without_tools():
    class DirectLLM:
        def chat_with_tools(self, messages, model, tools):
            return SimpleNamespace(content="Not covered in these videos.", tool_calls=None)

    ans = answer("unrelated?", "YC", FakeStore(), DirectLLM(), chat_model="m", top_k=5)
    assert ans.text.startswith("Not covered")
    assert ans.citations == []
