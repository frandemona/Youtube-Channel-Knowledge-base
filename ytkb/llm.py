from typing import Any


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
