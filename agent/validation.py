from __future__ import annotations

from agent.state import (
    Plan, Replan, Step, Verdict, CostReport, VideoMetadata, Conflict,
)
from agent.cost_model import segment_type_at, detect_unresolvable
from agent.planner_defaults import (
    DEFAULT_PLAN,
    DEFAULT_REPLAN_CACHE_FRESH, DEFAULT_REPLAN_CACHE_STALE,
    DEFAULT_REPLAN_NO_CAPTIONS, DEFAULT_REPLAN_TIE, DEFAULT_REPLAN_ABORT,
    ALLOWED_TOOLS_BY_FAILURE, FORBIDDEN_TOOLS_BY_FAILURE,
)


class PlanValidationError(Exception):
    pass


class ReplanValidationError(Exception):
    pass


class VerdictValidationError(Exception):
    pass


def validate_plan(plan: Plan) -> Plan:
    if not plan.steps:
        raise PlanValidationError("plan is empty")

    tool_names = [s.tool for s in plan.steps]
    if "export_video" not in tool_names:
        raise PlanValidationError("plan must contain export_video")
    if tool_names[-1] != "export_video":
        raise PlanValidationError("export_video must be the last step")

    if "detect_conflicts" in tool_names:
        idx = tool_names.index("detect_conflicts")
        if "fetch_caption_source" in tool_names and tool_names.index("fetch_caption_source") > idx:
            raise PlanValidationError("fetch_caption_source must precede detect_conflicts")
        if "fetch_audio_source" in tool_names and tool_names.index("fetch_audio_source") > idx:
            raise PlanValidationError("fetch_audio_source must precede detect_conflicts")

    if any(t not in {"fetch_caption_source", "fetch_audio_source",
                     "query_video_metadata", "query_metadata_cache",
                     "detect_conflicts", "export_video"} for t in tool_names):
        raise PlanValidationError("plan contains an out-of-vocabulary tool")

    if "evaluate_option_costs" in tool_names:
        raise PlanValidationError("evaluate_option_costs must NOT be a planned step; the decider owns it")

    return plan


def replan_for_cache_state(cache_state: dict) -> tuple[list[dict], str]:
    state = cache_state.get("state")
    if state == "valid" and cache_state.get("fresh", False):
        return DEFAULT_REPLAN_CACHE_FRESH, "fallback_cache"
    if state == "valid":
        return DEFAULT_REPLAN_CACHE_STALE, "fallback_stale_cache"
    if state in {"absent", "wrong_video_id", "signature_mismatch"}:
        return DEFAULT_REPLAN_NO_CAPTIONS, "export_no_captions"
    return DEFAULT_REPLAN_NO_CAPTIONS, "export_no_captions"


CANONICAL_PLAN_BY_STRATEGY = {
    "fallback_cache":      DEFAULT_REPLAN_CACHE_FRESH,
    "fallback_stale_cache": DEFAULT_REPLAN_CACHE_STALE,
    "export_no_captions":  DEFAULT_REPLAN_NO_CAPTIONS,
    "abort":               DEFAULT_REPLAN_ABORT,
}


def validate_replan_against_cache_state(replan: Replan, cache_state: dict) -> Replan:
    default_plan, expected_strategy = replan_for_cache_state(cache_state)
    canonical = CANONICAL_PLAN_BY_STRATEGY.get(expected_strategy, default_plan)
    canonical_tool_list = [s["tool"] for s in canonical]

    needs_correction = False
    reason = None

    replan_strategy = replan.strategy
    if replan_strategy != expected_strategy:
        needs_correction = True
        reason = f"LLM strategy '{replan_strategy}' != expected '{expected_strategy}' for cache_state.state='{cache_state.get('state')}'"

    if cache_state.get("state") in {"absent", "wrong_video_id", "signature_mismatch"}:
        if any(s.tool == "query_metadata_cache" for s in replan.new_plan):
            needs_correction = True
            reason = "LLM attempted to query untrustworthy cache"

    if cache_state.get("state") == "valid":
        if replan_strategy in {"fallback_cache", "fallback_stale_cache"}:
            llm_tool_list = [s.tool for s in replan.new_plan]
            if llm_tool_list != canonical_tool_list:
                needs_correction = True
                reason = "LLM tool list does not match canonical plan for this strategy"
    elif replan_strategy == "export_no_captions":
        llm_tool_list = [s.tool for s in replan.new_plan]
        if llm_tool_list != canonical_tool_list:
            needs_correction = True
            reason = "LLM tool list does not match canonical export_no_captions plan"

    forbidden = FORBIDDEN_TOOLS_BY_FAILURE.get("source_offline", set())
    if any(s.tool in forbidden for s in replan.new_plan):
        needs_correction = True
        reason = "LLM included a forbidden tool"

    if needs_correction:
        replan = Replan(
            new_plan=[Step(**s) for s in canonical],
            strategy=expected_strategy,
            rationale=replan.rationale if replan.rationale
                else f"deterministic replacement: {reason}",
            source=replan.source,
            corrected=True,
        )
        replan._correct_reason = reason
    return replan


def validate_replan_against_failure_mode(replan: Replan, failure_mode: str) -> Replan:
    forbidden = FORBIDDEN_TOOLS_BY_FAILURE.get(failure_mode, set())
    if any(s.tool in forbidden for s in replan.new_plan):
        if failure_mode == "unresolvable_tie":
            replacement = DEFAULT_REPLAN_TIE
            rep_strategy = "escalate_needs_review"
        else:
            replacement, rep_strategy = replan_for_cache_state({"state": "absent"})
        replan = Replan(
            new_plan=[Step(**s) for s in replacement],
            strategy=rep_strategy,
            rationale="LLM attempted a forbidden tool for this failure mode; corrected",
            source="fallback",
            corrected=True,
        )

    if failure_mode == "unresolvable_tie":
        tools = {s.tool for s in replan.new_plan}
        required = {"apply_tiebreaker", "export_video"}
        if tools != required:
            replan = Replan(
                new_plan=[Step(**s) for s in DEFAULT_REPLAN_TIE],
                strategy="escalate_needs_review",
                rationale="tie replan must be apply_tiebreaker + export_video only",
                source="fallback",
                corrected=True,
            )
        if replan.strategy != "escalate_needs_review":
            replan = Replan(
                new_plan=replan.new_plan,
                strategy="escalate_needs_review",
                rationale=replan.rationale,
                source=replan.source,
                corrected=True,
            )

    return replan


def validate_verdict_against_cost_report(
    verdict: Verdict,
    cost_report: CostReport,
    metadata: VideoMetadata,
    conflict: Conflict,
) -> Verdict:
    computed_tie = detect_unresolvable(cost_report, metadata, conflict)
    if verdict.llm_claimed_tie is None:
        verdict.llm_claimed_tie = verdict.tie
    verdict.tie = computed_tie

    valid_options = {o.option for o in cost_report.options}

    if computed_tie:
        if verdict.chosen_option is not None:
            verdict.corrected = True
        verdict.chosen_option = None
        verdict.confidence = "tie"
        return verdict

    if verdict.chosen_option is None:
        verdict.corrected = True
        verdict.chosen_option = cost_report.best
    elif verdict.chosen_option not in valid_options:
        verdict.corrected = True
        verdict.chosen_option = cost_report.best
    elif verdict.chosen_option != cost_report.best:
        verdict.corrected = True
        verdict.chosen_option = cost_report.best

    if verdict.confidence == "tie":
        verdict.confidence = "high"

    return verdict
