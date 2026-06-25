import json
from dataclasses import asdict

from .channel import ChannelInfo, ChannelFilters, resolve_channel, list_videos
from .config import Config
from .db import connect, upsert_video, videos_by_state, count_by_state, record_run, VideoState, set_state
from .embeddings import Embedder
from .llm import LLMClient
from .models import RunSummary, VideoMeta, Segment
from .paths import ChannelPaths, slugify, list_channel_slugs
from .pipeline import ChannelContext, process_video
from .store import ChannelStore
from .chunking import chunk_segments
from .transcripts import load_raw

# Videos interrupted mid-pipeline land in one of these intermediate states with no
# error; without re-including them, sync/retry would strand them forever.
INTERMEDIATE_STATES = [
    VideoState.TRANSCRIPT_FETCHED, VideoState.WHISPER_TRANSCRIBED, VideoState.AD_STRIPPED,
]
PROCESS_STATES = [
    VideoState.DISCOVERED, VideoState.FAILED_FETCH, VideoState.FAILED_EMBED,
] + INTERMEDIATE_STATES
RETRY_STATES = [
    VideoState.FAILED_FETCH, VideoState.FAILED_EMBED, VideoState.NO_TRANSCRIPT,
] + INTERMEDIATE_STATES


def save_channel(cfg: Config, slug: str, info: ChannelInfo, filters: ChannelFilters) -> None:
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    paths.ensure()
    paths.channel_json.write_text(json.dumps(
        {"info": asdict(info), "filters": asdict(filters)}, indent=2
    ))


def load_channel(cfg: Config, slug: str) -> tuple[ChannelInfo, ChannelFilters]:
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    data = json.loads(paths.channel_json.read_text())
    return ChannelInfo(**data["info"]), ChannelFilters(**data["filters"])


def add_channel(cfg, url, filters: ChannelFilters, name=None, *, resolver=None) -> str:
    resolver = resolver or resolve_channel
    info = resolver(url)
    slug = slugify(name or info.title or info.handle)
    save_channel(cfg, slug, info, filters)
    return slug


def build_context(cfg: Config, slug: str) -> ChannelContext:
    info, filters = load_channel(cfg, slug)
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    paths.ensure()
    conn = connect(paths.db)
    embedder = Embedder(cfg.embedding_model)
    store = ChannelStore(paths, conn, embedder)
    llm = LLMClient(cfg.openrouter_api_key) if cfg.openrouter_api_key else None
    return ChannelContext(paths=paths, conn=conn, cfg=cfg, llm=llm, store=store,
                          filters=filters, whisper_enabled=filters.whisper)


def _summarize(states: list[VideoState]) -> RunSummary:
    s = RunSummary()
    for st in states:
        if st == VideoState.INDEXED:
            s.done += 1
        elif st in (VideoState.NO_TRANSCRIPT,):
            s.skipped += 1
        else:
            s.failed += 1
    return s


def sync_channel(cfg, slug, *, dry_run=False, lister=None, process=None) -> RunSummary:
    lister = lister or list_videos
    process = process or process_video
    info, filters = load_channel(cfg, slug)
    ctx = build_context(cfg, slug)

    videos = lister(info.uploads_url, filters)
    summary = RunSummary(new=0)
    for meta in videos:
        if ctx.conn.execute("SELECT 1 FROM videos WHERE video_id=?", (meta.video_id,)).fetchone() is None:
            summary.new += 1
        upsert_video(ctx.conn, meta)

    if dry_run:
        record_run(ctx.conn, summary, "sync-dry")
        return summary

    todo = videos_by_state(ctx.conn, PROCESS_STATES)
    by_id = {m.video_id: m for m in videos}
    results = []
    for row in todo:
        meta = by_id.get(row["video_id"])
        if meta is None:
            meta = VideoMeta(row["video_id"], row["title"], row["duration"], row["upload_date"], row["url"])
        results.append(process(ctx, meta))
    s = _summarize(results)
    s.new = summary.new
    ctx.store.optimize()
    record_run(ctx.conn, s, "sync")
    return s


def retry_channel(cfg, slug, *, process=None) -> RunSummary:
    process = process or process_video
    info, filters = load_channel(cfg, slug)
    ctx = build_context(cfg, slug)
    todo = videos_by_state(ctx.conn, RETRY_STATES)
    results = []
    for row in todo:
        meta = VideoMeta(row["video_id"], row["title"], row["duration"], row["upload_date"], row["url"])
        results.append(process(ctx, meta))
    s = _summarize(results)
    ctx.store.optimize()
    record_run(ctx.conn, s, "retry")
    return s


def channel_status(cfg, slug) -> dict:
    paths = ChannelPaths.for_slug(cfg.data_dir, slug)
    conn = connect(paths.db)
    return {"slug": slug, "counts": count_by_state(conn)}


def list_channels(cfg) -> list[str]:
    return list_channel_slugs(cfg.data_dir)


def reindex_video(ctx: ChannelContext, row) -> VideoState | None:
    vid = row["video_id"]
    seg_path = ctx.paths.clean_segments_path(vid)
    if seg_path.exists():
        segments = load_raw(seg_path)
    elif ctx.paths.clean_path(vid).exists():
        # Legacy channels indexed before clean.json: text only, timestamps unavailable.
        segments = [Segment(0.0, 0.0, ctx.paths.clean_path(vid).read_text(encoding="utf-8"))]
    else:
        return None
    try:
        chunks = chunk_segments(vid, segments, ctx.cfg.chunk_tokens, ctx.cfg.chunk_overlap)
        if not chunks:
            return None
        ctx.store.add(chunks, title_of={vid: row["title"]})
    except Exception as e:
        set_state(ctx.conn, vid, VideoState.FAILED_EMBED, error=str(e))
        return VideoState.FAILED_EMBED
    return VideoState.INDEXED


def reindex_channel(cfg, slug, *, process=None) -> RunSummary:
    process = process or reindex_video
    ctx = build_context(cfg, slug)
    ctx.store.reset()
    summary = RunSummary()
    for row in videos_by_state(ctx.conn, [VideoState.INDEXED]):
        res = process(ctx, row)
        if res == VideoState.INDEXED:
            summary.done += 1
        elif res is None:
            summary.skipped += 1
        else:
            summary.failed += 1
    ctx.store.optimize()
    record_run(ctx.conn, summary, "reindex")
    return summary
