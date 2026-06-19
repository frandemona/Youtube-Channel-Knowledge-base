import requests

from .models import Segment

API = "https://api.sponsor.ajay.app/api/skipSegments"
CATEGORIES = ("sponsor", "selfpromo")


def _http_get(url: str, params: dict):
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_segments(video_id: str, *, http_get=None) -> list[tuple[float, float]]:
    http_get = http_get or _http_get
    data = http_get(
        "https://api.sponsor.ajay.app/api/skipSegments",
        {"videoID": video_id, "categories": '["sponsor","selfpromo"]'},
    )
    if not data:
        return []
    out = []
    for item in data:
        if item.get("category") in CATEGORIES:
            s, e = item["segment"]
            out.append((float(s), float(e)))
    return out


def strip_segments(segments: list[Segment], ad_ranges: list[tuple[float, float]]) -> list[Segment]:
    def in_ad(seg: Segment) -> bool:
        mid = (seg.start + seg.end) / 2
        return any(lo <= mid <= hi for lo, hi in ad_ranges)

    return [s for s in segments if not in_ad(s)]
