"""Tests for the deterministic cost model (the heart of the decision rule)."""
from __future__ import annotations

import pytest

from agent.state import Conflict, VideoMetadata, Segment
from agent.cost_model import (
    evaluate_option_costs, detect_unresolvable, segment_type_at,
    _drift_components,
)
from config import RERENDER_PENALTY, EPS_TIE, WEIGHTS


def _conflict(start_diff: float = 0.2, end_diff: float = 0.0,
              cap_start: float = 5.2, duck_start: float = 5.0):
    return Conflict(
        caption_id="c2", duck_id="d2",
        caption_start=cap_start, duck_start=duck_start,
        caption_end=cap_start + 2.8, duck_end=duck_start + 3.0,
        start_diff=start_diff, end_diff=end_diff,
        overlap_seconds=2.8, kind="start_misalignment",
    )


def _metadata(seg_type: str = "dialogue") -> VideoMetadata:
    return VideoMetadata(duration=12.0, segments=[
        Segment(start=0, end=4, type="dialogue"),
        Segment(start=4, end=8, type=seg_type),
        Segment(start=8, end=12, type="mixed"),
    ])


class TestCostModel:
    def test_dialogue_segment_prefers_trust_caption(self):
        cr = evaluate_option_costs(_conflict(), _metadata("dialogue"))
        assert cr.segment_type == "dialogue"
        assert cr.best == "trust_caption"
        trust_caption_cost = next(o.cost for o in cr.options if o.option == "trust_caption")
        trust_audio_cost = next(o.cost for o in cr.options if o.option == "trust_audio")
        assert trust_caption_cost < trust_audio_cost

    def test_music_segment_prefers_trust_audio(self):
        cr = evaluate_option_costs(_conflict(), _metadata("music"))
        assert cr.segment_type == "music"
        assert cr.best == "trust_audio"
        trust_caption_cost = next(o.cost for o in cr.options if o.option == "trust_caption")
        trust_audio_cost = next(o.cost for o in cr.options if o.option == "trust_audio")
        assert trust_audio_cost < trust_caption_cost

    def test_mixed_segment_is_tie(self):
        cr = evaluate_option_costs(_conflict(0.0), _metadata("mixed"))
        assert cr.segment_type == "mixed"
        assert detect_unresolvable(cr, _metadata("mixed"), _conflict(0.0)) is True

    def test_numeric_tie_when_small_drift_in_dialogue(self):
        c = _conflict(0.05, 0.0)
        cr = evaluate_option_costs(c, _metadata("dialogue"))
        assert cr.marginal_advantage < EPS_TIE
        assert detect_unresolvable(cr, _metadata("dialogue"), c) is True

    def test_rerender_never_wins_on_cost(self):
        cr = evaluate_option_costs(_conflict(), _metadata("dialogue"))
        re_option = next(o for o in cr.options if o.option == "re_render_duck")
        for o in cr.options:
            if o.option != "re_render_duck":
                assert o.cost < re_option.cost

    def test_drift_components_start_weighted_2x_end(self):
        start, end = _drift_components(_conflict(0.3, 0.4))
        assert start == 0.3
        assert end == 0.2
        assert start+end == 0.5

    def test_consistency_with_weights_table(self):
        for seg_type, weights in WEIGHTS.items():
            cr = evaluate_option_costs(_conflict(), _metadata(seg_type))
            assert cr.weights == weights

    def test_marginal_advantage_is_second_best_minus_best(self):
        cr = evaluate_option_costs(_conflict(), _metadata("dialogue"))
        srt = sorted(o.cost for o in cr.options)
        assert abs(cr.marginal_advantage - (srt[1] - srt[0])) < 1e-9

    def test_segment_type_at_returns_correct_segment_for_5p2(self):
        md = _metadata("dialogue")
        assert segment_type_at(md, 5.2) == "dialogue"
        md_music = _metadata("music")
        assert segment_type_at(md_music, 5.2) == "music"