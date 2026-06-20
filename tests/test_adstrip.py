from ytkb.adstrip import strip_ads
from ytkb.models import Segment
from ytkb.config import load_config


def segs():
    return [Segment(0, 5, "intro"), Segment(5, 10, "sponsor"), Segment(12, 20, "content")]


def test_sponsorblock_path(tmp_path):
    cfg = load_config(tmp_path)
    clean, method = strip_ads("v1", segs(), llm=None, cfg=cfg, sb_get=lambda vid: [(5.0, 11.0)])
    assert method == "sponsorblock"
    assert [s.text for s in clean] == ["intro", "content"]


class FakeLLM:
    def complete(self, messages, model):
        # returns the indices (0-based) of ad segments as CSV
        return "1"


def test_llm_fallback_when_no_sponsorblock(tmp_path):
    cfg = load_config(tmp_path)
    clean, method = strip_ads("v1", segs(), llm=FakeLLM(), cfg=cfg, sb_get=lambda vid: [])
    assert method == "llm"
    assert [s.text for s in clean] == ["intro", "content"]


def test_no_ads_returns_all(tmp_path):
    cfg = load_config(tmp_path)

    class NoAdLLM:
        def complete(self, messages, model):
            return ""

    clean, method = strip_ads("v1", segs(), llm=NoAdLLM(), cfg=cfg, sb_get=lambda vid: [])
    assert method == "none"
    assert len(clean) == 3
