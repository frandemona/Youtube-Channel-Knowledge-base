from ytkb.sponsorblock import get_segments, strip_segments
from ytkb.models import Segment


def test_strip_segments_drops_overlapping():
    segs = [
        Segment(0, 5, "intro"),
        Segment(5, 10, "sponsor read"),   # midpoint 7.5 in [5,12]
        Segment(12, 20, "content"),
    ]
    out = strip_segments(segs, [(5.0, 12.0)])
    assert [s.text for s in out] == ["intro", "content"]


def test_get_segments_parses_categories():
    api = [
        {"segment": [30.0, 45.0], "category": "sponsor"},
        {"segment": [60.0, 65.0], "category": "interaction"},  # ignored
        {"segment": [90.0, 95.0], "category": "selfpromo"},
    ]
    ranges = get_segments("v1", http_get=lambda url, params: api)
    assert ranges == [(30.0, 45.0), (90.0, 95.0)]


def test_get_segments_handles_no_data():
    assert get_segments("v1", http_get=lambda url, params: None) == []


def test_get_segments_swallows_network_errors():
    # SponsorBlock is best-effort: a network failure must degrade to "no data",
    # not raise (which would crash the whole sync).
    import requests

    def boom(url, params):
        raise requests.exceptions.ReadTimeout("slow")

    assert get_segments("v1", http_get=boom) == []
