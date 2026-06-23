from ytkb.paths import ChannelPaths
from ytkb.db import connect, upsert_video, set_state, VideoState
from ytkb.embeddings import Embedder
from ytkb.store import ChannelStore
from ytkb.models import VideoMeta, Chunk


class HashBackend:
    """Deterministic 8-dim embedding from word hashes so semantic search is testable."""
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 8
            for w in t.lower().split():
                v[hash(w) % 8] += 1.0
            out.append(v)
        return out


def build_store(tmp_path):
    paths = ChannelPaths.for_slug(tmp_path, "c")
    paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v1", "Cofounders", 600, "20240101", "https://youtu.be/v1"))
    set_state(conn, "v1", VideoState.INDEXED)
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    chunks = [
        Chunk("v1", 0, 10.0, "how to find a cofounder for your startup"),
        Chunk("v1", 1, 40.0, "pricing your product and revenue models"),
    ]
    store.add(chunks, title_of={"v1": "Cofounders"})
    return store


def test_keyword_search_finds_exact(tmp_path):
    store = build_store(tmp_path)
    hits = store.keyword_search("cofounder", k=5)
    assert hits[0].video_id == "v1"
    assert hits[0].start == 10.0


def test_semantic_search_returns_hits(tmp_path):
    store = build_store(tmp_path)
    hits = store.semantic_search("cofounder", k=1)
    assert len(hits) == 1
    assert hits[0].title == "Cofounders"


def test_list_videos(tmp_path):
    store = build_store(tmp_path)
    vids = store.list_videos()
    assert vids[0]["title"] == "Cofounders"


def test_add_is_idempotent_per_video(tmp_path):
    paths = ChannelPaths.for_slug(tmp_path, "c")
    paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v1", "Cofounders", 600, "20240101", "https://youtu.be/v1"))
    set_state(conn, "v1", VideoState.INDEXED)
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    chunks = [
        Chunk("v1", 0, 10.0, "how to find a cofounder for your startup"),
        Chunk("v1", 1, 40.0, "pricing your product and revenue models"),
    ]
    store.add(chunks, title_of={"v1": "Cofounders"})
    store.add(chunks, title_of={"v1": "Cofounders"})

    keyword_hits = store.keyword_search("cofounder", 50)
    assert len(keyword_hits) == 1

    semantic_hits = store.semantic_search("cofounder", k=1)
    assert len(semantic_hits) == 1

    # the vector table itself must not have accumulated a duplicate row per chunk
    assert store._table().count_rows() == len(chunks)


def test_read_around_missing_clean_file_no_crash(tmp_path):
    store = build_store(tmp_path)
    result = store.read_around("nonexistent_vid", None)
    assert result == ""


def test_reset_clears_index(tmp_path):
    store = build_store(tmp_path)
    assert store.keyword_search("cofounder", 5) != []
    assert store.semantic_search("cofounder", 5) != []
    store.reset()
    assert store.keyword_search("cofounder", 5) == []
    assert store.semantic_search("cofounder", 5) == []
    assert store._table() is None
    store.reset()  # idempotent: no error when already empty / table absent
