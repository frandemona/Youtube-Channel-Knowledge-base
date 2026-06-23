import requests

from .models import Segment

# Canonical SponsorBlock host (Cloudflare-fronted, reliable). The `api.` origin
# subdomain exists but is frequently slow/unreachable and caused read timeouts.
API = "https://sponsor.ajay.app/api/skipSegments"
CATEGORIES = ("sponsor", "selfpromo")


def _http_get(url: str, params: dict):
    # (connect, read) timeouts: fail reasonably fast if SponsorBlock is slow/unreachable.
    resp = requests.get(url, params=params, timeout=(5, 10))
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_segments(video_id: str, *, http_get=None) -> list[tuple[float, float]]:
    http_get = http_get or _http_get
    try:
        data = http_get(
            API,
            {"videoID": video_id, "categories": '["sponsor","selfpromo"]'},
        )
    except (requests.RequestException, ValueError):
        # SponsorBlock is best-effort: any network/parse failure means "no data",
        # so ad-stripping falls through to the LLM/keep-all path instead of crashing.
        return []
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
