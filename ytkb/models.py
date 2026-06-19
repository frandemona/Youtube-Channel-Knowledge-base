from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class VideoMeta:
    video_id: str
    title: str
    duration: int | None
    upload_date: str | None
    url: str


@dataclass
class Chunk:
    video_id: str
    idx: int
    start: float
    text: str


@dataclass
class ChunkHit:
    video_id: str
    title: str
    start: float
    text: str
    score: float


@dataclass
class Citation:
    video_id: str
    title: str
    start: float
    url: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]


@dataclass
class RunSummary:
    new: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
