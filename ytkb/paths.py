import re
from dataclasses import dataclass
from pathlib import Path


def slugify(name: str) -> str:
    name = name.strip().lower().lstrip("@")
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def channels_root(data_dir: Path) -> Path:
    return Path(data_dir) / "channels"


def list_channel_slugs(data_dir: Path) -> list[str]:
    root = channels_root(data_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@dataclass(frozen=True)
class ChannelPaths:
    dir: Path

    @classmethod
    def for_slug(cls, data_dir: Path, slug: str) -> "ChannelPaths":
        return cls(channels_root(data_dir) / slug)

    @property
    def channel_json(self) -> Path:
        return self.dir / "channel.json"

    @property
    def db(self) -> Path:
        return self.dir / "videos.db"

    @property
    def transcripts_dir(self) -> Path:
        return self.dir / "transcripts"

    @property
    def vectors_dir(self) -> Path:
        return self.dir / "vectors"

    def raw_path(self, video_id: str) -> Path:
        return self.transcripts_dir / f"{video_id}.raw.json"

    def clean_path(self, video_id: str) -> Path:
        return self.transcripts_dir / f"{video_id}.clean.txt"

    def ensure(self) -> None:
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
