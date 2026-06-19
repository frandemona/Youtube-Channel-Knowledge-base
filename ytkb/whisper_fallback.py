import tempfile
from pathlib import Path

from yt_dlp import YoutubeDL

from .models import Segment


def _download_audio(video_id: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    out = tmp / f"{video_id}.%(ext)s"
    opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": str(out),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([f"https://youtu.be/{video_id}"])
    return next(tmp.glob(f"{video_id}.*"))


def transcribe(video_id: str, model_size: str, *, audio_path_fn=None, model=None) -> list[Segment]:
    audio_path_fn = audio_path_fn or _download_audio
    if model is None:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="auto", compute_type="int8")
    path = audio_path_fn(video_id)
    segments, _info = model.transcribe(str(path))
    return [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]
