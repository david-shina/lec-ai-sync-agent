"""Tests for the deterministic tiebreaker policy."""
from __future__ import annotations

import pytest

from agent.tiebreaker import apply_tiebreaker, TIEBREAKER_POLICY
from agent.state import CostReport, CostOption


def _cr(opts):
    sorted_opts = sorted([CostOption(**o) for o in opts], key=lambda x:(x.cost, x.option))
    return CostReport(
        conflict_id="c2/d2", segment_type="mixed",
        weights={"w_caption":0.7, "w_audio":0.7, "w_rerender":1.0},
        options=sorted_opts, best=sorted_opts[0].option,
        marginal_advantage=0.0,
    )


class TestTiebreaker:
    def test_prefers_zero_rerender_when_tied(self):
        cr = _cr([
            {"option":"trust_caption","cost":0.14,"breakdown":{"rerender":0}},
            {"option":"trust_audio","cost":0.14,"breakdown":{"rerender":0}},
            {"option":"re_render_duck","cost":8.0,"breakdown":{"rerender":1.0}},
        ])
        result = apply_tiebreaker(cr)
        assert result["chosen"] == "trust_audio"
        assert result["policy"] == TIEBREAKER_POLICY
        assert result["confidence"] == "tie"

    def test_both_tied_options_recorded(self):
        cr = _cr([
            {"option":"trust_caption","cost":0.14,"breakdown":{"rerender":0}},
            {"option":"trust_audio","cost":0.14,"breakdown":{"rerender":0}},
            {"option":"re_render_duck","cost":8.0,"breakdown":{"rerender":1.0}},
        ])
        result = apply_tiebreaker(cr)
        assert len(result["both_options"]) == 2
        assert all(o["breakdown"]["rerender"] == 0 for o in result["both_options"])
        assert all(o["breakdown"]["rerender"] > 0 for o in result["rejected_options"])

    def test_re_render_duck_never_wins_tiebreaker(self):
        cr = _cr([
            {"option":"trust_caption","cost":0.14,"breakdown":{"rerender":0}},
            {"option":"trust_audio","cost":0.14,"breakdown":{"rerender":0}},
            {"option":"re_render_duck","cost":0.14,"breakdown":{"rerender":1.0}},
        ])
        result = apply_tiebreaker(cr)
        assert result["chosen"] != "re_render_duck"