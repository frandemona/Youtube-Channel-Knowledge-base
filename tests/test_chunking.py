from ytkb.chunking import chunk_segments
from ytkb.models import Segment


def test_chunks_respect_token_budget_and_timestamps():
    segs = [Segment(start=float(i * 5), end=float(i * 5 + 5), text="word " * 10) for i in range(5)]
    chunks = chunk_segments("v1", segs, chunk_tokens=20, overlap=0)
    assert chunks[0].idx == 0
    assert chunks[0].start == 0.0
    assert all(len(c.text.split()) <= 20 for c in chunks)
    # 50 words / 20 per chunk -> 3 chunks
    assert len(chunks) == 3


def test_overlap_repeats_words():
    segs = [Segment(0, 5, "a b c d e f g h")]
    chunks = chunk_segments("v1", segs, chunk_tokens=4, overlap=2)
    assert chunks[0].text.split() == ["a", "b", "c", "d"]
    assert chunks[1].text.split()[:2] == ["c", "d"]


def test_empty_segments():
    assert chunk_segments("v1", [], 100, 0) == []
