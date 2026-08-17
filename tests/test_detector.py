"""Tests for conflict detector: pairing heuristic + tolerances."""
from __future__ import annotations

import pytest

from agent.detector import detect_conflicts, _intervals_should_pair, _overlap
from agent.state import Cue, Duck


class TestDetector:
    def test_detects_start_misalignment(self):
        caps = [{"id":"c1","text":"hello","start":5.2,"end":8.0}]
        ducks = [{"id":"d1","start":5.0,"end":8.0,"gain_reduction_db":-12}]
        conflicts = detect_conflicts(caps, ducks)
        assert len(conflicts) == 1
        assert conflicts[0].kind == "start_misalignment"
        assert conflicts[0].start_diff == pytest.approx(0.2)

    def test_detects_end_misalignment(self):
        caps = [{"id":"c1","text":"hello","start":5.0,"end":8.2}]
        ducks = [{"id":"d1","start":5.0,"end":8.0,"gain_reduction_db":-12}]
        conflicts = detect_conflicts(caps, ducks)
        assert len(conflicts) == 1
        assert conflicts[0].kind == "end_misalignment"

    def test_no_conflict_when_aligned_within_tolerance(self):
        caps = [{"id":"c1","text":"hello","start":5.05,"end":8.0}]
        ducks = [{"id":"d1","start":5.0,"end":8.0,"gain_reduction_db":-12}]
        assert detect_conflicts(caps, ducks) == []

    def test_no_conflict_when_intervals_do_not_pair(self):
        caps = [{"id":"c1","text":"hello","start":1.0,"end":2.0}]
        ducks = [{"id":"d1","start":10.0,"end":11.0,"gain_reduction_db":-12}]
        assert detect_conflicts(caps, ducks) == []

    def test_overlap_helper(self):
        assert _overlap(1,3,2,4) == 1.0
        assert _overlap(1,2,3,4) == 0.0
        assert _overlap(1,4,1,4) == 3.0

    def test_pairing_helper_positive(self):
        cap = Cue(id="c", text="x", start=5.0, end=8.0)
        duck = Duck(id="d", start=5.0, end=8.0)
        assert _intervals_should_pair(cap, duck)

    def test_pairing_helper_negative_far_apart(self):
        cap = Cue(id="c", text="x", start=5.0, end=8.0)
        far = Duck(id="d2", start=50.0, end=53.0)
        assert not _intervals_should_pair(cap, far)