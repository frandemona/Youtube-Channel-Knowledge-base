from types import SimpleNamespace
from ytkb.llm import LLMClient, ToolCallStream


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tc(index, id=None, name=None, args=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=args))


class FakeStreamingClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured = {}

    class _Chat:
        def __init__(self, outer): self.outer = outer
        @property
        def completions(self): return FakeStreamingClient._Comp(self.outer)

    class _Comp:
        def __init__(self, outer): self.outer = outer
        def create(self, model, messages, tools, tool_choice, stream):
            self.outer.captured = {"stream": stream, "tool_choice": tool_choice}
            return iter(self.outer._chunks)

    @property
    def chat(self): return FakeStreamingClient._Chat(self)


def test_stream_yields_content_deltas():
    chunks = [_chunk(content="Hello "), _chunk(content="world")]
    llm = LLMClient(api_key="k", client=FakeStreamingClient(chunks))
    stream = llm.stream_with_tools([{"role": "user", "content": "x"}], model="m", tools=[])
    assert list(stream) == ["Hello ", "world"]
    assert stream.tool_calls == []


def test_stream_accumulates_tool_calls_across_chunks():
    chunks = [
        _chunk(tool_calls=[_tc(0, id="call1", name="keyword_search", args='{"que')]),
        _chunk(tool_calls=[_tc(0, args='ry":"cofounder"}')]),
    ]
    llm = LLMClient(api_key="k", client=FakeStreamingClient(chunks))
    stream = llm.stream_with_tools([], model="m", tools=[{"x": 1}])
    assert list(stream) == []  # no content
    tc = stream.tool_calls[0]
    assert tc.id == "call1"
    assert tc.function.name == "keyword_search"
    assert tc.function.arguments == '{"query":"cofounder"}'
