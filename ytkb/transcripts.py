import json
from pathlib import Path

from yt_dlp import YoutubeDL

from .models import Segment


def parse_json3(data: dict) -> list[Segment]:
    out: list[Segment] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        start = ev["tStartMs"] / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        out.append(Segment(start=start, end=start + dur, text=text))
    return out


def _yt_dlp_download(video_id: str, languages: list[str]) -> dict | None:
    opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "json3",
        "subtitleslangs": languages,
    }
    url = f"https://youtu.be/{video_id}"
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    subs = {**(info.get("subtitles") or {}), **(info.get("automatic_captions") or {})}
    for lang in languages:
        tracks = subs.get(lang) or subs.get(lang.split("-")[0])
        if not tracks:
            continue
        for t in tracks:
            if t.get("ext") == "json3":
                data = ydl.urlopen(t["url"]).read()
                return json.loads(data)
    return None


def fetch_captions(video_id: str, languages: list[str], *, downloader=None) -> list[Segment] | None:
    downloader = downloader or _yt_dlp_download
    raw = downloader(video_id, languages)
    if raw is None:
        return None
    segs = parse_json3(raw)
    return segs or None


def save_raw(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.__dict__ for s in segments], ensure_ascii=False))


def load_raw(path: Path) -> list[Segment]:
    data = json.loads(Path(path).read_text())
    return [Segment(**d) for d in data]
