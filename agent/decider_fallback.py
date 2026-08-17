from __future__ import annotations

from agent.state import Verdict, CostReport, VideoMetadata, Conflict
from agent.cost_model import detect_unresolvable


def _conflict_id(c: Conflict) -> str:
    return f"{c.caption_id}/{c.duck_id}"


def deterministic_verdicts(
    conflicts: list[Conflict],
    cost_reports_by_conflict: dict[str, CostReport],
    metadata: VideoMetadata,
) -> list[Verdict]:
    out: list[Verdict] = []
    for c in conflicts:
        cid = _conflict_id(c)
        cr = cost_reports_by_conflict[cid]
        computed_tie = detect_unresolvable(cr, metadata, c)
        chosen = None if computed_tie else cr.best
        seg_phrase = {"dialogue": "dialogue segment",
                      "music": "music segment",
                      "mixed": "mixed segment"}.get(cr.segment_type, cr.segment_type)
        v = Verdict(
            conflict_id=cid,
            chosen_option=chosen,
            rationale=(
                f"Conflict {cid} in {seg_phrase}. CostReport best={cr.best}, "
                f"marginal_advantage={cr.marginal_advantage}. Computed tie={computed_tie}. "
                f"Deterministic fallback applied; LLM reasoning unavailable."
            ),
            cost_breakdown={o.option: {"cost": o.cost, **o.breakdown} for o in cr.options},
            confidence="tie" if computed_tie else "high",
            tie=computed_tie,
            source="deterministic_fallback",
            llm_claimed_tie=None,
            corrected=False,
        )
        out.append(v)
    return out
