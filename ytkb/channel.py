from contextlib import contextmanager
from dataclasses import dataclass

from yt_dlp import YoutubeDL

from .models import VideoMeta


@dataclass
class ChannelInfo:
    channel_id: str
    handle: str
    title: str
    uploads_url: str


@dataclass
class ChannelFilters:
    max: int | None = None
    since: str | None = None      # YYYYMMDD
    min_length: int | None = None
    no_shorts: bool = False
    whisper: bool = True


@contextmanager
def _default_ydl(extra: dict | None = None):
    opts = {"quiet": True, "skip_download": True, "extract_flat": "in_playlist"}
    if extra:
        opts.update(extra)
    with YoutubeDL(opts) as ydl:
        yield ydl


def resolve_channel(url: str, *, ydl=None) -> ChannelInfo:
    if ydl is None:
        with _default_ydl() as real:
            info = real.extract_info(url, download=False)
    else:
        info = ydl.extract_info(url, download=False)
    channel_id = info.get("channel_id") or info.get("uploader_id")
    title = info.get("channel") or info.get("uploader") or channel_id
    handle = info.get("uploader_id") or info.get("channel_id") or ""
    return ChannelInfo(
        channel_id=channel_id,
        handle=handle,
        title=title,
        uploads_url=f"https://www.youtube.com/channel/{channel_id}/videos",
    )


def _passes_filters(meta: VideoMeta, f: ChannelFilters) -> bool:
    dur = meta.duration or 0
    if f.no_shorts and dur and dur < 60:
        return False
    if f.min_length and dur and dur < f.min_length:
        return False
    if f.since and meta.upload_date and meta.upload_date < f.since:
        return False
    return True


def list_videos(uploads_url: str, filters: ChannelFilters, *, ydl=None) -> list[VideoMeta]:
    if ydl is None:
        with _default_ydl() as real:
            info = real.extract_info(uploads_url, download=False)
    else:
        info = ydl.extract_info(uploads_url, download=False)
    out: list[VideoMeta] = []
    for e in info.get("entries", []) or []:
        meta = VideoMeta(
            video_id=e["id"],
            title=e.get("title") or e["id"],
            duration=e.get("duration"),
            upload_date=e.get("upload_date"),
            url=f"https://youtu.be/{e['id']}",
        )
        if _passes_filters(meta, filters):
            out.append(meta)
        if filters.max and len(out) >= filters.max:
            break
    return out
