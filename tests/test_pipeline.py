from ytkb.paths import ChannelPaths
from ytkb.db import connect, upsert_video, get_video, VideoState
from ytkb.embeddings import Embedder
from ytkb.store import ChannelStore
from ytkb.channel import ChannelFilters
from ytkb.pipeline import ChannelContext, process_video
from ytkb.models import VideoMeta, Segment
from ytkb.config import load_config
from ytkb.transcripts import load_raw
from tests.test_store import HashBackend


def make_ctx(tmp_path, whisper_enabled=True):
    paths = ChannelPaths.for_slug(tmp_path, "c")
    paths.ensure()
    conn = connect(paths.db)
    cfg = load_config(tmp_path)
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    return ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=None, store=store,
                          filters=ChannelFilters(), whisper_enabled=whisper_enabled)


def meta(vid="v1"):
    return VideoMeta(vid, "Title", 600, "20240101", f"https://youtu.be/{vid}")


def test_pipeline_happy_path(tmp_path):
    ctx = make_ctx(tmp_path)
    upsert_video(ctx.conn, meta())
    segs = [Segment(0, 5, "find a cofounder"), Segment(5, 10, "this is sponsored by acme")]
    state = process_video(
        ctx, meta(),
        fetch=lambda vid, langs: segs,
        strip=lambda vid, s, llm, cfg, **k: ([segs[0]], "llm"),
    )
    assert state == VideoState.INDEXED
    assert get_video(ctx.conn, "v1")["state"] == "indexed"
    assert ctx.paths.clean_path("v1").exists()
    assert ctx.store.keyword_search("cofounder", 5)[0].video_id == "v1"


def test_pipeline_no_captions_whisper_disabled(tmp_path):
    ctx = make_ctx(tmp_path, whisper_enabled=False)
    upsert_video(ctx.conn, meta())
    state = process_video(ctx, meta(), fetch=lambda vid, langs: None)
    assert state == VideoState.NO_TRANSCRIPT


def test_pipeline_whisper_fallback(tmp_path):
    ctx = make_ctx(tmp_path, whisper_enabled=True)
    upsert_video(ctx.conn, meta())
    segs = [Segment(0, 5, "whisper text about cofounder")]
    state = process_video(
        ctx, meta(),
        fetch=lambda vid, langs: None,
        whisper=lambda vid, model: segs,
        strip=lambda vid, s, llm, cfg, **k: (s, "none"),
    )
    assert state == VideoState.INDEXED


def test_pipeline_writes_timestamped_clean_json(tmp_path):
    ctx = make_ctx(tmp_path)
    upsert_video(ctx.conn, meta())
    segs = [Segment(10.0, 12.0, "find a cofounder"), Segment(12.0, 15.0, "this is sponsored")]
    process_video(
        ctx, meta(),
        fetch=lambda vid, langs: segs,
        strip=lambda vid, s, llm, cfg, **k: ([segs[0]], "llm"),
    )
    clean_json = ctx.paths.clean_segments_path("v1")
    assert clean_json.exists()
    loaded = load_raw(clean_json)
    assert loaded == [Segment(10.0, 12.0, "find a cofounder")]  # cleaned + timestamp preserved


def test_pipeline_adstrip_failure_sets_failed_embed(tmp_path):
    # A failure in the ad-strip step (e.g. SponsorBlock/LLM error) must mark just
    # this video FAILED_EMBED (retryable), not crash the whole sync.
    ctx = make_ctx(tmp_path)
    upsert_video(ctx.conn, meta())
    segs = [Segment(0, 5, "hello")]

    def boom_strip(vid, s, llm, cfg, **k):
        raise RuntimeError("sponsorblock down")

    state = process_video(ctx, meta(), fetch=lambda vid, langs: segs, strip=boom_strip)
    assert state == VideoState.FAILED_EMBED
    assert get_video(ctx.conn, "v1")["state"] == "failed_embed"


def test_pipeline_fetch_error_falls_through_to_whisper(tmp_path):
    # A caption-fetch error (e.g. HTTP 429 on an auto-translated track) must fall
    # through to Whisper when enabled, not dead-end at FAILED_FETCH.
    ctx = make_ctx(tmp_path, whisper_enabled=True)
    upsert_video(ctx.conn, meta())
    segs = [Segment(0, 5, "english audio transcribed by whisper")]

    def boom_fetch(vid, langs):
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    state = process_video(ctx, meta(), fetch=boom_fetch,
                          whisper=lambda vid, model: segs,
                          strip=lambda vid, s, llm, cfg, **k: (s, "none"))
    assert state == VideoState.INDEXED
    assert get_video(ctx.conn, "v1")["state"] == "indexed"


def test_pipeline_fetch_error_whisper_disabled_is_failed_fetch(tmp_path):
    # With Whisper disabled, a caption-fetch error stays retryable (FAILED_FETCH).
    ctx = make_ctx(tmp_path, whisper_enabled=False)
    upsert_video(ctx.conn, meta())

    def boom_fetch(vid, langs):
        raise RuntimeError("HTTP Error 429")

    state = process_video(ctx, meta(), fetch=boom_fetch)
    assert state == VideoState.FAILED_FETCH
    assert "429" in get_video(ctx.conn, "v1")["last_error"]
