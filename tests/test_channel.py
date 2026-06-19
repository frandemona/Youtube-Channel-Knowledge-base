from ytkb.channel import _passes_filters, ChannelFilters, list_videos, resolve_channel
from ytkb.models import VideoMeta


def vm(vid, dur=600, date="20240115", title="t"):
    return VideoMeta(video_id=vid, title=title, duration=dur, upload_date=date, url=f"https://youtu.be/{vid}")


def test_filters_no_shorts():
    f = ChannelFilters(no_shorts=True)
    assert _passes_filters(vm("a", dur=600), f) is True
    assert _passes_filters(vm("b", dur=30), f) is False


def test_filters_min_length_and_since():
    f = ChannelFilters(min_length=300, since="20240101")
    assert _passes_filters(vm("a", dur=600, date="20240115"), f) is True
    assert _passes_filters(vm("b", dur=100, date="20240115"), f) is False  # too short
    assert _passes_filters(vm("c", dur=600, date="20231201"), f) is False  # too old


class FakeYDL:
    def __init__(self, info):
        self._info = info

    def extract_info(self, url, download=False):
        return self._info


def test_resolve_channel_from_video_url():
    info = {"channel_id": "UC123", "uploader_id": "@yc", "channel": "Y Combinator"}
    ci = resolve_channel("https://youtu.be/abc", ydl=FakeYDL(info))
    assert ci.channel_id == "UC123"
    assert ci.handle == "@yc"
    assert ci.uploads_url == "https://www.youtube.com/channel/UC123/videos"


def test_list_videos_applies_max_and_filters():
    entries = {
        "entries": [
            {"id": "a", "title": "Long", "duration": 600, "upload_date": "20240115"},
            {"id": "b", "title": "Short", "duration": 30, "upload_date": "20240115"},
            {"id": "c", "title": "Long2", "duration": 700, "upload_date": "20240110"},
        ]
    }
    vids = list_videos("u", ChannelFilters(no_shorts=True, max=1), ydl=FakeYDL(entries))
    assert [v.video_id for v in vids] == ["a"]  # short filtered, max=1
