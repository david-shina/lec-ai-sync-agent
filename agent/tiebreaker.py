"""Deterministic tiebreaker policy for unresolvable tie conflicts."""
from __future__ import annotations

from agent.state import CostReport


TIEBREAKER_POLICY = "min_rerender_then_audio_continuity"


def apply_tiebreaker(cost_report: CostReport) -> dict:
    """Apply tiebreaker to a single conflict cost report."""
    zero_rerender = [o for o in cost_report.options if o.breakdown.get("rerender", 0) == 0]

    bucket = zero_rerender if zero_rerender else cost_report.options
    bucket_sorted = sorted(
        bucket,
        key=lambda o: (o.cost, 0 if o.option == "trust_audio" else 1),
    )
    winner = bucket_sorted[0].option if bucket_sorted else "trust_audio"

    return {
        "conflict_id": cost_report.conflict_id,
        "chosen": winner,
        "policy": TIEBREAKER_POLICY,
        "both_options": [o.model_dump() for o in zero_rerender],
        "rejected_options": [
            o.model_dump() for o in cost_report.options if o.breakdown.get("rerender", 0) > 0
        ],
        "confidence": "tie",
    }
