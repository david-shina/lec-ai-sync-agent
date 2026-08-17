from __future__ import annotations

from agent.state import Cue, Duck, Conflict
from config import EPS_START, EPS_END, PAIR_TOLERANCE


def _intervals_should_pair(cap: Cue | dict, duck: Duck | dict, tolerance: float = PAIR_TOLERANCE) -> bool:
    """Determines if two intervals should be paired based on their start and end times."""
    if isinstance(cap, dict):
        cap = Cue(**cap)
    if isinstance(duck, dict):
        duck = Duck(**duck)

    start_close = abs(cap.start - duck.start) < tolerance
    if start_close:
        return True

    overlap = min(cap.end, duck.end) - max(cap.start, duck.start)
    if overlap <= 0:
        return False

    span = max(cap.end - cap.start, duck.end - duck.start, 0.001)
    return overlap / span > 0.5


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Returns the overlap between two intervals [a_start, a_end] and [b_start, b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def detect_conflicts(captions: list, ducks: list) -> list[Conflict]:
    """
    Detects conflicts between captions and ducks based on their start and end times."""
    cues = [c if isinstance(c, Cue) else Cue(**c) for c in captions]
    windows = [d if isinstance(d, Duck) else Duck(**d) for d in ducks]

    conflicts: list[Conflict] = []
    for cap in cues:
        for duck in windows:
            if not _intervals_should_pair(cap, duck):
                continue

            start_diff = cap.start - duck.start
            end_diff = cap.end - duck.end
            overlap = _overlap(cap.start, cap.end, duck.start, duck.end)

            kind = None
            if abs(start_diff) > EPS_START and abs(end_diff) > EPS_END:
                kind = "both_misaligned"
            elif abs(start_diff) > EPS_START:
                kind = "start_misalignment"
            elif abs(end_diff) > EPS_END:
                kind = "end_misalignment"

            if kind is None:
                continue

            conflicts.append(
                Conflict(
                    caption_id=cap.id,
                    duck_id=duck.id,
                    caption_start=cap.start,
                    duck_start=duck.start,
                    caption_end=cap.end,
                    duck_end=duck.end,
                    start_diff=round(start_diff, 6),
                    end_diff=round(end_diff, 6),
                    overlap_seconds=round(overlap, 6),
                    kind=kind,
                )
            )
    return conflicts
