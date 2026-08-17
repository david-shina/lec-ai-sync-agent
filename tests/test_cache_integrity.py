"""Tests for cache integrity (signature mismatch detection)."""
from __future__ import annotations

import json
import os
import tempfile
import pytest

from agent.cache_integrity import (
    compute_video_signature, write_cache, query_cache_state, cache_file_for,
)
from config import CACHE_DIR


class TestCacheIntegrity:
    def test_signature_changes_with_file_content(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f1:
            f1.write(b"A" * 200 * 1024); f1.flush(); p1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f2:
            f2.write(b"B" * 200 * 1024); f2.flush(); p2 = f2.name
        try:
            assert compute_video_signature(p1) != compute_video_signature(p2)
        finally:
            os.unlink(p1); os.unlink(p2)

    def test_signature_stable_for_same_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"X" * 200 * 1024); f.flush(); p = f.name
        try:
            assert compute_video_signature(p) == compute_video_signature(p)
        finally:
            os.unlink(p)

    def test_query_absent_cache(self):
        cf = cache_file_for("nonexistent_video_for_test")
        if cf.exists():
            cf.unlink()
        state = query_cache_state("nonexistent_video_for_test", "/dev/null")
        assert state["state"] == "absent"

    def test_write_then_query_valid_cache_fresh(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf:
            vf.write(b"C" * 200 * 1024); vf.flush(); vpath = vf.name
        try:
            vid = "test_cache_integrity_v1"
            cf = cache_file_for(vid)
            if cf.exists():
                cf.unlink()
            write_cache(vid, vpath, [{"id":"c1","text":"x","start":1.0,"end":2.0}])
            cs = query_cache_state(vid, vpath)
            assert cs["state"] == "valid"
            assert cs["fresh"] is True
            assert cs["captions"] == [{"id":"c1","text":"x","start":1.0,"end":2.0}]
        finally:
            os.unlink(vpath)

    def test_signature_mismatch_detected_when_video_changes(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf1:
            vf1.write(b"D" * 200 * 1024); vf1.flush(); vpath1 = vf1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf2:
            vf2.write(b"E" * 250 * 1024); vf2.flush(); vpath2 = vf2.name
        try:
            vid = "test_cache_integrity_v2"
            cf = cache_file_for(vid)
            if cf.exists():
                cf.unlink()
            write_cache(vid, vpath1, [{"id":"c1","text":"x","start":1.0,"end":2.0}])
            cs = query_cache_state(vid, vpath2)
            assert cs["state"] == "signature_mismatch"
        finally:
            os.unlink(vpath1); os.unlink(vpath2)

    def test_wrong_video_id_detected(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf:
            vf.write(b"F" * 200 * 1024); vf.flush(); vpath = vf.name
        try:
            vid = "test_cache_integrity_v3"
            cf = cache_file_for(vid)
            if cf.exists():
                cf.unlink()
            write_cache(vid, vpath, [{"id":"c1","text":"x","start":1.0,"end":2.0}])
            payload = json.loads(cf.read_text(encoding="utf-8"))
            payload["video_id"] = "different_video"
            cf.write_text(json.dumps(payload), encoding="utf-8")
            cs = query_cache_state(vid, vpath)
            assert cs["state"] == "wrong_video_id"
        finally:
            os.unlink(vpath)