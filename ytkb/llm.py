from dataclasses import dataclass
from typing import Any


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str | None
    function: _Fn


class ToolCallStream:
    """Iterate to get content-delta strings; after iteration, `.tool_calls` is populated."""

    def __init__(self, raw_stream):
        self._raw = raw_stream
        self.tool_calls: list[ToolCall] = []

    def __iter__(self):
        acc: dict[int, dict] = {}
        for chunk in self._raw:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = acc.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
        self.tool_calls = [
            ToolCall(id=s["id"], function=_Fn(name=s["name"], arguments=s["args"]))
            for _, s in sorted(acc.items())
        ]


class LLMClient:
    def __init__(self, api_key: str | None, base_url: str = "https://openrouter.ai/api/v1", *, client=None):
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[dict], model: str) -> str:
        resp = self._client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    def chat_with_tools(self, messages: list[dict], model: str, tools: list[dict]) -> Any:
        resp = self._client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto"
        )
        return resp.choices[0].message

    def stream_with_tools(self, messages: list[dict], model: str, tools: list[dict]) -> "ToolCallStream":
        raw = self._client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto", stream=True
        )
        return ToolCallStream(raw)
