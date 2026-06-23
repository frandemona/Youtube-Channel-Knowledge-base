from .models import Segment, Chunk


def chunk_segments(video_id: str, segments: list[Segment], chunk_tokens: int, overlap: int) -> list[Chunk]:
    # Build a flat list of (word, start_time) so each chunk knows its first word's timestamp.
    words: list[tuple[str, float]] = []
    for seg in segments:
        for w in seg.text.split():
            words.append((w, seg.start))
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_tokens - overlap)
    idx = 0
    i = 0
    while i < len(words):
        window = words[i : i + chunk_tokens]
        text = " ".join(w for w, _ in window)
        chunks.append(Chunk(video_id=video_id, idx=idx, start=window[0][1], text=text))
        idx += 1
        if i + chunk_tokens >= len(words):
            break
        i += step
    return chunks
