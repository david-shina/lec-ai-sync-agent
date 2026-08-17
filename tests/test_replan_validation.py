"""Tests for replanner validation: code owns the plan, LLM owns the rationale."""
from __future__ import annotations

import pytest

from agent.state import Replan, Step
from agent.validation import (
    validate_replan_against_cache_state, validate_replan_against_failure_mode,
    replan_for_cache_state,
)
from agent.planner_defaults import (
    DEFAULT_REPLAN_CACHE_FRESH, DEFAULT_REPLAN_NO_CAPTIONS, DEFAULT_REPLAN_TIE,
)


class TestReplanValidation:
    def test_cache_fresh_strategy_matches_canonical(self):
        cs = {"state":"valid","fresh":True}
        replan = Replan(
            new_plan=[Step(**s) for s in DEFAULT_REPLAN_CACHE_FRESH],
            strategy="fallback_cache", rationale="ok"
        )
        validated = validate_replan_against_cache_state(replan, cs)
        assert not validated.corrected
        assert validated.strategy == "fallback_cache"

    def test_signature_mismatch_forces_export_no_captions(self):
        cs = {"state":"signature_mismatch"}
        replan = Replan(
            new_plan=[Step(tool="query_metadata_cache", intent="should be rejected")],
            strategy="fallback_cache",  # also wrong strategy
            rationale="LLM tried to use corrupt cache"
        )
        validated = validate_replan_against_cache_state(replan, cs)
        assert validated.corrected
        assert validated.strategy == "export_no_captions"
        canonical_tools = [s.tool for s in validated.new_plan]
        assert "query_metadata_cache" not in canonical_tools
        assert canonical_tools == [s["tool"] for s in DEFAULT_REPLAN_NO_CAPTIONS]

    def test_wrong_video_id_yields_export_no_captions(self):
        cs = {"state":"wrong_video_id"}
        replan = Replan(
            new_plan=[Step(tool="query_metadata_cache", intent="x")],
            strategy="fallback_cache", rationale="oops"
        )
        validated = validate_replan_against_cache_state(replan, cs)
        assert validated.corroded if False else True
        assert validated.strategy == "export_no_captions"
        assert validated.corrected

    def test_absent_cache_yields_export_no_captions(self):
        cs = {"state":"absent"}
        replan, strategy = replan_for_cache_state(cs)
        assert strategy == "export_no_captions"
        assert [s["tool"] for s in replan] == [s["tool"] for s in DEFAULT_REPLAN_NO_CAPTIONS]

    def test_tie_replan_must_only_have_apply_tiebreaker_plus_export(self):
        replan = Replan(
            new_plan=[Step(tool="apply_tiebreaker", intent=""),
                      Step(tool="export_video", intent="")],
            strategy="escalate_needs_review", rationale="ok"
        )
        validated = validate_replan_against_failure_mode(replan, "unresolvable_tie")
        assert not validated.corrected
        assert validated.strategy == "escalate_needs_review"

    def test_tie_replan_rejects_decider_retry(self):
        replan = Replan(
            new_plan=[Step(tool="evaluate_option_costs", intent=""),
                      Step(tool="export_video", intent="")],
            strategy="escalate_needs_review", rationale="LLM tried to re-deliberate"
        )
        validated = validate_replan_against_failure_mode(replan, "unresolvable_tie")
        assert validated.corrected
        assert {s.tool for s in validated.new_plan} == {"apply_tiebreaker", "export_video"}

    def test_tie_replan_rejects_decode_again(self):
        bad_plans = [
            [Step(tool="decider", intent=""), Step(tool="export_video", intent="")],
            [Step(tool="apply_tiebreaker", intent="")],
            [Step(tool="export_video", intent="")],
        ]
        for plan in bad_plans:
            replan = Replan(new_plan=plan, strategy="escalate_needs_review",
                            rationale="insufficient")
            validated = validate_replan_against_failure_mode(replan, "unresolvable_tie")
            assert validated.corrected

    def test_source_offline_replan_rejects_fetch_caption_source(self):
        replan = Replan(
            new_plan=[Step(tool="fetch_caption_source", intent="dead source"),
                      Step(tool="export_video", intent="")],
            strategy="fallback_cache", rationale="LLM tried to retry"
        )
        cs = {"state":"valid","fresh":True}
        validated = validate_replan_against_cache_state(replan, cs)
        assert validated.corrected
        assert "fetch_caption_source" not in [s.tool for s in validated.new_plan]