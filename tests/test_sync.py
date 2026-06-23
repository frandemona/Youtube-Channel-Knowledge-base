from ytkb.config import load_config
from ytkb.channel import ChannelInfo, ChannelFilters
from ytkb.db import VideoState, get_video
from ytkb.models import VideoMeta
from ytkb import sync


def info():
    return ChannelInfo("UC1", "@yc", "Y Combinator", "https://www.youtube.com/channel/UC1/videos")


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
