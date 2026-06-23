import json


def _hits_to_json(hits) -> str:
    return json.dumps([
        {"video_id": h.video_id, "title": h.title, "start": h.start, "text": h.text}
        for h in hits
    ])


def build_tools(store, top_k: int):
    def _spec(name, desc, props, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    specs = [
        _spec("keyword_search", "Exact keyword/BM25 search over transcript chunks. Best for names, jargon, exact phrases.",
              {"query": {"type": "string"}}, ["query"]),
        _spec("semantic_search", "Semantic search over transcript chunks. Best for concepts and paraphrases.",
              {"query": {"type": "string"}}, ["query"]),
        _spec("read_transcript", "Read transcript text for a video, optionally around a timestamp (seconds).",
              {"video_id": {"type": "string"}, "around_ts": {"type": "number"}}, ["video_id"]),
        _spec("list_videos", "List indexed videos; optionally filter by a substring of the title.",
              {"contains": {"type": "string"}}, []),
    ]

    def dispatch(name: str, args: dict) -> str:
        if name == "keyword_search":
            return _hits_to_json(store.keyword_search(args["query"], top_k))
        if name == "semantic_search":
            return _hits_to_json(store.semantic_search(args["query"], top_k))
        if name == "read_transcript":
            if not args.get("video_id"):
                return json.dumps({"error": "read_transcript requires video_id"})
            return store.read_around(args["video_id"], args.get("around_ts"), args.get("window", 90.0))
        if name == "list_videos":
            return json.dumps(store.list_videos(args.get("contains")))
        return json.dumps({"error": f"unknown tool {name}"})

    return specs, dispatch
