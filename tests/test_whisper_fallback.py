from types import SimpleNamespace
from ytkb.whisper_fallback import transcribe


class FakeModel:
    def transcribe(self, path):
        segs = [
            SimpleNamespace(start=0.0, end=1.0, text="hello"),
            SimpleNamespace(start=1.0, end=2.0, text=" world"),
        ]
        return segs, SimpleNamespace(language="en")


def test_transcribe_maps_segments(tmp_path):
    audio = tmp_path / "v1.m4a"
    audio.write_bytes(b"x")
    out = transcribe("v1", "base", audio_path_fn=lambda vid: audio, model=FakeModel())
    assert [(s.start, s.text) for s in out] == [(0.0, "hello"), (1.0, "world")]
