from types import SimpleNamespace
from ytkb.llm import LLMClient


class FakeChat:
    def __init__(self, content):
        self._content = content
        self.calls = []

    class _Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, model, messages, **kwargs):
            self.outer.calls.append({"model": model, "messages": messages, **kwargs})
            msg = SimpleNamespace(content=self.outer._content, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    @property
    def completions(self):
        return FakeChat._Completions(self)


class FakeClient:
    def __init__(self, content):
        self.chat = FakeChat(content)


def test_complete_returns_text():
    llm = LLMClient(api_key="k", client=FakeClient("hi there"))
    out = llm.complete([{"role": "user", "content": "x"}], model="m")
    assert out == "hi there"
