from ytkb.config import load_config
from ytkb.channel import ChannelInfo, ChannelFilters
from ytkb.db import VideoState, get_video
from ytkb.models import VideoMeta
from ytkb import sync


def info():
    return ChannelInfo("UC1", "@yc", "Y Combinator", "https://www.youtube.com/channel/UC1/videos")


def _indexed_ctx_with_clean_json(tmp_path):
    """Build a channel whose video 'v1' is INDEXED and has a timestamped clean.json on disk."""
    from ytkb.paths import ChannelPaths
    from ytkb.db import connect, upsert_video, set_state
    from ytkb.embeddings import Embedder
    from ytkb.store import ChannelStore
    from ytkb.pipeline import ChannelContext
    from ytkb.transcripts import save_raw
    from ytkb.models import Segment
    from tests.test_store import HashBackend

    cfg = load_config(tmp_path)
    paths = ChannelPaths.for_slug(cfg.data_dir, "c")
    paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v1", "Cofounders", 600, "20240101", "https://youtu.be/v1"))
    set_state(conn, "v1", VideoState.INDEXED)
    save_raw(paths.clean_segments_path("v1"), [Segment(10.0, 12.0, "find a cofounder")])
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    ctx = ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=None, store=store,
                         filters=ChannelFilters(), whisper_enabled=True)
    return ctx


def test_add_and_load_channel(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "https://youtu.be/x", ChannelFilters(no_shorts=True),
                            resolver=lambda url, **k: info())
    assert slug == "y-combinator"
    ci, filters = sync.load_channel(cfg, slug)
    assert ci.channel_id == "UC1"
    assert filters.no_shorts is True


def test_sync_processes_new_videos(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    vids = [VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a"),
            VideoMeta("b", "B", 600, "20240102", "https://youtu.be/b")]
    processed = []

    def fake_process(ctx, meta, **k):
        processed.append(meta.video_id)
        from ytkb.db import set_state
        set_state(ctx.conn, meta.video_id, VideoState.INDEXED)
        return VideoState.INDEXED

    summary = sync.sync_channel(cfg, slug, lister=lambda url, f, **k: vids, process=fake_process)
    assert sorted(processed) == ["a", "b"]
    assert summary.new == 2 and summary.done == 2


def test_sync_dry_run_processes_nothing(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    vids = [VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a")]
    called = []
    summary = sync.sync_channel(cfg, slug, dry_run=True, lister=lambda url, f, **k: vids,
                                process=lambda *a, **k: called.append(1))
    assert called == []
    assert summary.new == 1


def test_sync_skips_already_indexed(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    vids = [VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a")]

    def fake_process(ctx, meta, **k):
        from ytkb.db import set_state
        set_state(ctx.conn, meta.video_id, VideoState.INDEXED)
        return VideoState.INDEXED

    sync.sync_channel(cfg, slug, lister=lambda url, f, **k: vids, process=fake_process)
    count = []
    sync.sync_channel(cfg, slug, lister=lambda url, f, **k: vids,
                      process=lambda ctx, meta, **k: count.append(1))
    assert count == []  # already indexed -> not reprocessed


def test_sync_reprocesses_prior_failed_not_in_listing(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    vids = [VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a")]

    def fake_process_fail(ctx, meta, **k):
        from ytkb.db import set_state
        set_state(ctx.conn, meta.video_id, VideoState.FAILED_FETCH)
        return VideoState.FAILED_FETCH

    sync.sync_channel(cfg, slug, lister=lambda url, f, **k: vids, process=fake_process_fail)

    processed_ids = []

    def fake_process_record(ctx, meta, **k):
        processed_ids.append(meta.video_id)
        from ytkb.db import set_state
        set_state(ctx.conn, meta.video_id, VideoState.INDEXED)
        return VideoState.INDEXED

    sync.sync_channel(cfg, slug, lister=lambda url, f, **k: [], process=fake_process_record)
    assert "a" in processed_ids


def test_reindex_video_rebuilds_from_clean_json(tmp_path):
    ctx = _indexed_ctx_with_clean_json(tmp_path)
    row = get_video(ctx.conn, "v1")
    state = sync.reindex_video(ctx, row)
    assert state == VideoState.INDEXED
    hits = ctx.store.keyword_search("cofounder", 5)
    assert hits[0].video_id == "v1"
    assert hits[0].start == 10.0  # timestamp preserved from clean.json


def test_reindex_video_skips_when_no_clean_files(tmp_path):
    from ytkb.paths import ChannelPaths
    from ytkb.db import connect, upsert_video, set_state
    from ytkb.embeddings import Embedder
    from ytkb.store import ChannelStore
    from ytkb.pipeline import ChannelContext
    from tests.test_store import HashBackend
    cfg = load_config(tmp_path)
    paths = ChannelPaths.for_slug(cfg.data_dir, "c"); paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v9", "No files", 600, "20240101", "https://youtu.be/v9"))
    set_state(conn, "v9", VideoState.INDEXED)
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    ctx = ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=None, store=store,
                         filters=ChannelFilters(), whisper_enabled=True)
    assert sync.reindex_video(ctx, get_video(conn, "v9")) is None


def test_reindex_video_empty_transcript_is_skipped(tmp_path):
    from ytkb.paths import ChannelPaths
    from ytkb.db import connect, upsert_video, set_state, get_video
    from ytkb.embeddings import Embedder
    from ytkb.store import ChannelStore
    from ytkb.pipeline import ChannelContext
    from ytkb.transcripts import save_raw
    from tests.test_store import HashBackend
    cfg = load_config(tmp_path)
    paths = ChannelPaths.for_slug(cfg.data_dir, "c"); paths.ensure()
    conn = connect(paths.db)
    upsert_video(conn, VideoMeta("v1", "Empty", 600, "20240101", "https://youtu.be/v1"))
    set_state(conn, "v1", VideoState.INDEXED)
    save_raw(paths.clean_segments_path("v1"), [])  # cleaned transcript has no segments
    store = ChannelStore(paths, conn, Embedder("fake", backend=HashBackend()))
    ctx = ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=None, store=store,
                         filters=ChannelFilters(), whisper_enabled=True)
    assert sync.reindex_video(ctx, get_video(conn, "v1")) is None
    assert store.keyword_search("anything", 5) == []


def test_reindex_channel_resets_processes_indexed_only_and_logs(tmp_path):
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    ctx = sync.build_context(cfg, slug)
    from ytkb.db import upsert_video, set_state, connect
    from ytkb.paths import ChannelPaths
    upsert_video(ctx.conn, VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a"))
    set_state(ctx.conn, "a", VideoState.INDEXED)
    upsert_video(ctx.conn, VideoMeta("b", "B", 600, "20240102", "https://youtu.be/b"))
    set_state(ctx.conn, "b", VideoState.FAILED_FETCH)  # must NOT be reindexed

    processed = []
    def fake_process(ctx, row):
        processed.append(row["video_id"]); return VideoState.INDEXED

    summary = sync.reindex_channel(cfg, slug, process=fake_process)
    assert processed == ["a"]
    assert summary.done == 1
    # a "reindex" run was logged
    conn2 = connect(ChannelPaths.for_slug(cfg.data_dir, slug).db)
    kinds = [r["kind"] for r in conn2.execute("SELECT kind FROM runs").fetchall()]
    assert "reindex" in kinds


def test_retry_reprocesses_stranded_intermediate_state(tmp_path):
    # A video interrupted mid-pipeline (e.g. transcript_fetched, no error) must be
    # recovered by retry/sync, not stranded forever.
    from ytkb.db import upsert_video, set_state
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    ctx = sync.build_context(cfg, slug)
    upsert_video(ctx.conn, VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a"))
    set_state(ctx.conn, "a", VideoState.TRANSCRIPT_FETCHED)

    processed = []
    def fake_process(ctx, meta, **k):
        processed.append(meta.video_id)
        set_state(ctx.conn, meta.video_id, VideoState.INDEXED)
        return VideoState.INDEXED

    summary = sync.retry_channel(cfg, slug, process=fake_process)
    assert "a" in processed
    assert summary.done == 1


def test_sync_reprocesses_stranded_intermediate_state(tmp_path):
    from ytkb.db import upsert_video, set_state
    cfg = load_config(tmp_path)
    slug = sync.add_channel(cfg, "u", ChannelFilters(), resolver=lambda url, **k: info())
    ctx = sync.build_context(cfg, slug)
    upsert_video(ctx.conn, VideoMeta("a", "A", 600, "20240101", "https://youtu.be/a"))
    set_state(ctx.conn, "a", VideoState.AD_STRIPPED)  # stranded after ad-strip

    processed = []
    def fake_process(ctx, meta, **k):
        processed.append(meta.video_id)
        set_state(ctx.conn, meta.video_id, VideoState.INDEXED)
        return VideoState.INDEXED

    # empty listing: the stranded video isn't in the channel listing, only in the DB
    sync.sync_channel(cfg, slug, lister=lambda url, f, **k: [], process=fake_process)
    assert "a" in processed
