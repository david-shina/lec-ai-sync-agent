from __future__ import annotations

from agent.state import Conflict, VideoMetadata, CostReport, CostOption
from config import WEIGHTS, RERENDER_PENALTY, EPS_TIE


def _drift_components(conflict: Conflict) -> tuple[float, float]:
    """
    Returns the start and end drift components of a conflict.  Where the start drift is the absolute value of the start difference, 
    and the end drift is half the absolute value of the end difference. Because a caption that appears late is more perceptually jarring than one that lingers a bit long"""
    start = abs(conflict.start_diff)
    end = 0.5 * abs(conflict.end_diff)
    return start, end


def _evaluate_one(option: str, conflict: Conflict, weights: dict[str, float]) -> CostOption:
    """
    Evaluates the cost of a single option for resolving a conflict, given the conflict and the weights for the segment type. 
     Returns a CostOption object with the option name, total cost, and breakdown of costs.
    """
    start, end = _drift_components(conflict)
    drift_total = start + end

    if option == "trust_caption":
        caption_drift = 0.0
        audio_drift = drift_total
        rerender = 0.0
    elif option == "trust_audio":
        caption_drift = drift_total
        audio_drift = 0.0
        rerender = 0.0
    elif option == "re_render_duck":
        caption_drift = 0.0
        audio_drift = 0.0
        rerender = 1.0
    else:
        raise ValueError(f"unknown option {option!r}")

    cost = (
        weights["w_caption"] * caption_drift
        + weights["w_audio"] * audio_drift
        + weights["w_rerender"] * RERENDER_PENALTY * rerender
    )

    return CostOption(
        option=option,
        cost=round(cost, 6),
        breakdown={
            "caption_drift": round(caption_drift, 6),
            "audio_drift": round(audio_drift, 6),
            "rerender": rerender,
            "drift_components": {"start": round(start, 6), "end": round(end, 6)},
        },
    )


def evaluate_option_costs(conflict: Conflict, metadata: VideoMetadata) -> CostReport:
    """
    Evaluates the costs of all options for resolving a conflict, given the conflict and the video metadata. 
    Returns a CostReport object with the conflict id, segment type, weights, options, best option, and marginal advantage."""
    seg_type = segment_type_at(metadata, conflict.caption_start)
    weights = WEIGHTS.get(seg_type, WEIGHTS["mixed"])

    option_names = ["trust_caption", "trust_audio", "re_render_duck"]
    options = [_evaluate_one(name, conflict, weights) for name in option_names]
    options.sort(key=lambda o: (o.cost, o.option))

    best = options[0].option
    sorted_costs = sorted(o.cost for o in options)
    marg = round(sorted_costs[1] - sorted_costs[0], 6)

    return CostReport(
        conflict_id=f"{conflict.caption_id}/{conflict.duck_id}",
        segment_type=seg_type,
        weights=weights,
        options=options,
        best=best,
        marginal_advantage=marg,
    )


def segment_type_at(metadata: VideoMetadata, t: float) -> str:
    """
    Returns the segment type at a given time t in the video metadata. If no segment is found, returns "mixed"."""
    for seg in metadata.segments:
        if seg.start <= t <= seg.end:
            return seg.type
    return "mixed"


def detect_unresolvable(cost_report: CostReport, metadata: VideoMetadata, conflict: Conflict) -> bool:
    """
    Detects if a conflict is unresolvable based on the cost report and video metadata.
    A conflict is considered unresolvable if the marginal advantage of the best option is less than EPS_TIE, 
    `or if the segment type at the caption start time is "mixed".
    """
    seg_type = segment_type_at(metadata, conflict.caption_start)
    return (cost_report.marginal_advantage < EPS_TIE) or (seg_type == "mixed")
