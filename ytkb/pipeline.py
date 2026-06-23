from dataclasses import dataclass

from . import transcripts, whisper_fallback, adstrip
from .channel import ChannelFilters
from .chunking import chunk_segments
from .config import Config
from .db import set_state, set_ad_method, VideoState
from .models import VideoMeta
from .store import ChannelStore


@dataclass
class ChannelContext:
    paths: object
    conn: object
    cfg: Config
    llm: object
    store: ChannelStore
    filters: ChannelFilters
    whisper_enabled: bool


def process_video(ctx: ChannelContext, meta: VideoMeta, *, fetch=None, whisper=None, strip=None) -> VideoState:
    fetch = fetch or transcripts.fetch_captions
    whisper = whisper or whisper_fallback.transcribe
    strip = strip or adstrip.strip_ads
    vid = meta.video_id

    # 1. transcript
    try:
        segments = fetch(vid, ctx.cfg.languages)
    except Exception as e:  # network / extraction failure -> retryable
        set_state(ctx.conn, vid, VideoState.FAILED_FETCH, error=str(e))
        return VideoState.FAILED_FETCH

    if segments:
        set_state(ctx.conn, vid, VideoState.TRANSCRIPT_FETCHED)
    elif ctx.whisper_enabled:
        try:
            segments = whisper(vid, ctx.cfg.whisper_model)
        except Exception as e:
            set_state(ctx.conn, vid, VideoState.FAILED_FETCH, error=f"whisper: {e}")
            return VideoState.FAILED_FETCH
        if not segments:
            set_state(ctx.conn, vid, VideoState.NO_TRANSCRIPT, error="whisper produced nothing")
            return VideoState.NO_TRANSCRIPT
        set_state(ctx.conn, vid, VideoState.WHISPER_TRANSCRIBED)
    else:
        set_state(ctx.conn, vid, VideoState.NO_TRANSCRIPT, error="no captions; whisper disabled")
        return VideoState.NO_TRANSCRIPT

    transcripts.save_raw(ctx.paths.raw_path(vid), segments)

    # 2. ad stripping
    try:
        clean_segments, method = strip(vid, segments, ctx.llm, ctx.cfg)
    except Exception as e:
        set_state(ctx.conn, vid, VideoState.FAILED_EMBED, error=f"adstrip: {e}")
        return VideoState.FAILED_EMBED
    set_ad_method(ctx.conn, vid, method)
    ctx.paths.clean_path(vid).write_text(
        " ".join(s.text for s in clean_segments), encoding="utf-8"
    )
    transcripts.save_raw(ctx.paths.clean_segments_path(vid), clean_segments)
    set_state(ctx.conn, vid, VideoState.AD_STRIPPED)

    # 3. chunk + index
    try:
        chunks = chunk_segments(vid, clean_segments, ctx.cfg.chunk_tokens, ctx.cfg.chunk_overlap)
        ctx.store.add(chunks, title_of={vid: meta.title})
    except Exception as e:
        set_state(ctx.conn, vid, VideoState.FAILED_EMBED, error=str(e))
        return VideoState.FAILED_EMBED

    set_state(ctx.conn, vid, VideoState.INDEXED)
    return VideoState.INDEXED
