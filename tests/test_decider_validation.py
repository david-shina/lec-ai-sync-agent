"""Tests for validate_verdict_against_cost_report: code overrides LLM verdicts."""
from __future__ import annotations

import pytest

from agent.state import Verdict, CostReport, CostOption, Conflict, VideoMetadata, Segment
from agent.validation import validate_verdict_against_cost_report


def _conflict():
    return Conflict(
        caption_id="c2", duck_id="d2",
        caption_start=5.2, duck_start=5.0,
        caption_end=8.0, duck_end=8.0,
        start_diff=0.2, end_diff=0.0,
        overlap_seconds=2.8, kind="start_misalignment",
    )


def _cr_dialogue():
    return CostReport(
        conflict_id="c2/d2", segment_type="dialogue",
        weights={"w_caption":1.0, "w_audio":0.4, "w_rerender":1.0},
        options=[
            CostOption(option="trust_caption", cost=0.08, breakdown={}),
            CostOption(option="trust_audio",   cost=0.20, breakdown={}),
            CostOption(option="re_render_duck",cost=8.0,  breakdown={}),
        ],
        best="trust_caption", marginal_advantage=0.12,
    )


def _cr_tie():
    return CostReport(
        conflict_id="c2/d2", segment_type="mixed",
        weights={"w_caption":0.7, "w_audio":0.7, "w_rerender":1.0},
        options=[
            CostOption(option="trust_caption", cost=0.14, breakdown={}),
            CostOption(option="trust_audio",   cost=0.14, breakdown={}),
            CostOption(option="re_render_duck",cost=8.0,  breakdown={}),
        ],
        best="trust_caption", marginal_advantage=0.0,
    )


def _md对话():
    return VideoMetadata(duration=12.0, segments=[
        Segment(start=0, end=4, type="dialogue"),
        Segment(start=4, end=8, type="dialogue"),
        Segment(start=8, end=12, type="mixed"),
    ])


def _md_mixed_at_5():
    return VideoMetadata(duration=12.0, segments=[
        Segment(start=0, end=4, type="dialogue"),
        Segment(start=4, end=8, type="mixed"),
        Segment(start=8, end=12, type="dialogue"),
    ])


class TestVerdictValidation:
    def test_accepts_llm_pick_when_matches_cost_best(self):
        v = Verdict(conflict_id="c2/d2", chosen_option="trust_caption", rationale="ok",
                    confidence="medium", tie=False)
        validated = validate_verdict_against_cost_report(v, _cr_dialogue(), _md对话(), _conflict())
        assert not validated.corrected
        assert validated.chosen_option == "trust_caption"
        assert validated.tie is False

    def test_corrects_llm_pick_when_disagrees_with_cost_best(self):
        v = Verdict(conflict_id="c2/d2", chosen_option="trust_audio", rationale="bad call", tie=False)
        validated = validate_verdict_against_cost_report(v, _cr_dialogue(), _md对话(), _conflict())
        assert validated.corrected
        assert validated.chosen_option == "trust_caption"
        assert validated.tie is False

    def test_rejects_hallucinated_option_not_in_cost_report(self):
        v = Verdict(conflict_id="c2/d2", chosen_option="keep_both", rationale="made up", tie=False)
        validated = validate_verdict_against_cost_report(v, _cr_dialogue(), _md对话(), _conflict())
        assert validated.corrected
        assert validated.chosen_option == "trust_caption"

    def test_force_tie_when_marginal_below_epsilon(self):
        cr_with_small_margin = CostReport(
            conflict_id="c2/d2", segment_type="dialogue",
            weights={"w_caption":1.0, "w_audio":0.4, "w_rerender":1.0},
            options=[
                CostOption(option="trust_caption", cost=0.02, breakdown={}),
                CostOption(option="trust_audio",   cost=0.03, breakdown={}),
                CostOption(option="re_render_duck",cost=8.0, breakdown={}),
            ],
            best="trust_caption", marginal_advantage=0.01,
        )
        v = Verdict(conflict_id="c2/d2", chosen_option="trust_caption", rationale="clean", tie=False)
        validated = validate_verdict_against_cost_report(v, cr_with_small_margin, _md对话(), _conflict())
        assert validated.tie is True
        assert validated.chosen_option is None
        assert validated.confidence == "tie"

    def test_force_tie_in_mixed_segment_regardless_of_llm_claim(self):
        v = Verdict(conflict_id="c2/d2", chosen_option="trust_caption",
                    rationale="i think its a clear winner", tie=False)
        validated = validate_verdict_against_cost_report(v, _cr_tie(), _md_mixed_at_5(), _conflict())
        assert validated.tie is True
        assert validated.chosen_option is None

    def test_preserves_llm_claimed_tie_for_audit(self):
        v = Verdict(conflict_id="c2/d2", chosen_option=None, rationale="iam not sure", tie=True)
        validated = validate_verdict_against_cost_report(v, _cr_tie(), _md_mixed_at_5(), _conflict())
        assert validated.llm_claimed_tie is True
        assert validated.tie is True