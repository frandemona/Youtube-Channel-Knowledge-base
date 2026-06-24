import json
from dataclasses import asdict

from .models import Answer, Citation
from .retrieval import build_tools

SYSTEM = (
    "You are an expert assistant answering questions using ONLY the indexed videos from the "
    "YouTube channel '{channel}'. Use the search tools to find relevant transcript passages, "
    "read more context if needed, then answer concisely. Always ground claims in what the "
    "videos actually say. If the videos do not cover the question, say so plainly. "
    "Do as few searches as needed."
)


def answer(question, channel_title, store, llm, *, chat_model, top_k, max_steps=5) -> Answer:
    specs, dispatch = build_tools(store, top_k)
    messages = [
        {"role": "system", "content": SYSTEM.format(channel=channel_title)},
        {"role": "user", "content": question},
    ]
    cited: dict[str, Citation] = {}

    for _ in range(max_steps):
        msg = llm.chat_with_tools(messages, model=chat_model, tools=specs)
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return Answer(text=msg.content or "", citations=list(cited.values()))

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = dispatch(tc.function.name, args)
            _collect_citations(tc.function.name, result, cited)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # max_steps hit: ask for a final answer with no tools
    msg = llm.chat_with_tools(messages + [
        {"role": "user", "content": "Give your best final answer now using what you found."}
    ], model=chat_model, tools=[])
    return Answer(text=msg.content or "", citations=list(cited.values()))


def _collect_citations(tool_name: str, result: str, cited: dict) -> None:
    if tool_name not in ("keyword_search", "semantic_search"):
        return
    try:
        rows = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return
    for r in rows:
        vid, start = r.get("video_id"), r.get("start", 0.0)
        key = f"{vid}:{int(start)}"
        if vid and key not in cited:
            cited[key] = Citation(
                video_id=vid, title=r.get("title", ""), start=float(start),
                url=f"https://youtu.be/{vid}?t={int(start)}",
            )


def _status_for(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json or "{}")
    except (json.JSONDecodeError, TypeError):
        args = {}
    if name in ("keyword_search", "semantic_search"):
        q = args.get("query", "")
        return f"Searching transcripts for '{q}'…" if q else "Searching transcripts…"
    if name == "read_transcript":
        return f"Reading {args.get('video_id', 'a video')}…"
    if name == "list_videos":
        return "Listing videos…"
    return "Working…"


def answer_stream(question, channel_title, store, llm, *, chat_model, top_k, history=None, max_steps=5):
    try:
        specs, dispatch = build_tools(store, top_k)
        messages = [{"role": "system", "content": SYSTEM.format(channel=channel_title)}]
        messages += history or []
        messages.append({"role": "user", "content": question})
        cited: dict[str, Citation] = {}

        answered = False
        for _ in range(max_steps):
            stream = llm.stream_with_tools(messages, chat_model, specs)
            content = ""
            for token in stream:
                content += token
                yield {"type": "token", "text": token}
            tool_calls = stream.tool_calls
            if not tool_calls:
                answered = True
                break
            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                yield {"type": "status", "text": _status_for(tc.function.name, tc.function.arguments)}
                args = json.loads(tc.function.arguments or "{}")
                result = dispatch(tc.function.name, args)
                _collect_citations(tc.function.name, result, cited)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if not answered:
            # max_steps exhausted with tools still pending: force a final tool-less answer
            final = llm.stream_with_tools(
                messages + [{"role": "user", "content": "Give your best final answer now using what you found."}],
                chat_model, [])
            for token in final:
                yield {"type": "token", "text": token}

        yield {"type": "citations", "citations": [asdict(c) for c in cited.values()]}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "text": str(e)}
        yield {"type": "done"}


TITLE_PROMPT = (
    "Write a short 3-6 word title (no quotes, no trailing punctuation) summarizing this "
    "conversation.\n\nUser: {q}\nAssistant: {a}\n\nTitle:"
)


def generate_title(llm, model: str, question: str, answer: str) -> str:
    title = ""
    try:
        raw = llm.complete(
            [{"role": "user", "content": TITLE_PROMPT.format(q=question, a=answer[:500])}], model
        )
        title = (raw or "").strip().strip('"').splitlines()[0].strip()[:60] if raw and raw.strip() else ""
    except Exception:
        title = ""
    if not title:
        title = " ".join(question.split()[:6])
    return title
