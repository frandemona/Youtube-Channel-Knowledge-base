from pathlib import Path
from ytkb.paths import slugify, ChannelPaths, channels_root, list_channel_slugs


def test_slugify_normalizes():
    assert slugify("Y Combinator!") == "y-combinator"
    assert slugify("@MyChannel") == "mychannel"


def test_channel_paths_layout(tmp_path: Path):
    cp = ChannelPaths.for_slug(tmp_path, "ycombinator")
    assert cp.dir == tmp_path / "channels" / "ycombinator"
    assert cp.channel_json == cp.dir / "channel.json"
    assert cp.db == cp.dir / "videos.db"
    assert cp.raw_path("abc123") == cp.transcripts_dir / "abc123.raw.json"
    assert cp.clean_path("abc123") == cp.transcripts_dir / "abc123.clean.txt"


def test_list_channel_slugs(tmp_path: Path):
    (channels_root(tmp_path) / "a").mkdir(parents=True)
    (channels_root(tmp_path) / "b").mkdir(parents=True)
    assert list_channel_slugs(tmp_path) == ["a", "b"]
