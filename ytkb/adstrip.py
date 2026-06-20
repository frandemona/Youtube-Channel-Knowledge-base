from .config import Config
from .models import Segment
from .sponsorblock import get_segments as sb_get_segments, strip_segments

PROMPT = (
    "You are removing in-video sponsor/advertising reads from a transcript. "
    "Below are numbered transcript segments. Return ONLY a comma-separated list of the "
    "segment numbers (0-based) that are sponsor reads, ads, or self-promotion. "
    "If there are none, return an empty string.\n\n"
)


def _llm_ad_indices(segments: list[Segment], llm, model: str) -> set[int]:
    numbered = "\n".join(f"{i}: {s.text}" for i, s in enumerate(segments))
    raw = llm.complete([{"role": "user", "content": PROMPT + numbered}], model=model)
    idxs: set[int] = set()
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit():
            idxs.add(int(tok))
    return idxs


def strip_ads(video_id, segments, llm, cfg: Config, *, sb_get=None) -> tuple[list[Segment], str]:
    sb_get = sb_get or sb_get_segments
    ad_ranges = sb_get(video_id)
    if ad_ranges:
        return strip_segments(segments, ad_ranges), "sponsorblock"
    if llm is None:
        return list(segments), "none"
    idxs = _llm_ad_indices(segments, llm, cfg.adstrip_model)
    if not idxs:
        return list(segments), "none"
    return [s for i, s in enumerate(segments) if i not in idxs], "llm"
