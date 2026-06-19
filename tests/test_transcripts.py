from ytkb.transcripts import parse_json3, fetch_captions, save_raw, load_raw
from ytkb.models import Segment

JSON3 = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "hello "}, {"utf8": "world"}]},
        {"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "next line"}]},
        {"tStartMs": 4000, "segs": [{"utf8": "\n"}]},  # whitespace-only -> dropped
    ]
}


def test_parse_json3():
    segs = parse_json3(JSON3)
    assert segs == [
        Segment(start=0.0, end=2.0, text="hello world"),
        Segment(start=2.0, end=3.5, text="next line"),
    ]


def test_fetch_captions_returns_none_when_missing():
    assert fetch_captions("v1", ["en"], downloader=lambda vid, langs: None) is None


def test_fetch_captions_parses():
    segs = fetch_captions("v1", ["en"], downloader=lambda vid, langs: JSON3)
    assert segs[0].text == "hello world"


def test_save_and_load_roundtrip(tmp_path):
    segs = [Segment(0.0, 2.0, "a"), Segment(2.0, 3.0, "b")]
    p = tmp_path / "v1.raw.json"
    save_raw(p, segs)
    assert load_raw(p) == segs
