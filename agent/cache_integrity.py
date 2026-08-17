from __future__ import annotations

import hashlib
import os
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from config import CACHE_DIR, STALE_THRESHOLD_HOURS


def compute_video_signature(video_path: str) -> str:
    size = os.path.getsize(video_path)
    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))

    with open(video_path, "rb") as f:
        head = f.read(64 * 1024)
        h.update(head)
        if size > 128 * 1024:
            f.seek(-64 * 1024, os.SEEK_END)
            tail = f.read(64 * 1024)
            h.update(tail)

    return h.hexdigest()


def _hours_since(ts_iso: str) -> float:
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except Exception:
        return float("inf")
    now = datetime.now(timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 3600.0)


def cache_file_for(video_id: str) -> Path:
    return CACHE_DIR / f"captions_{video_id}.json"


def write_cache(video_id: str, video_path: str, captions: list) -> None:
    payload = {
        "video_id": video_id,
        "video_signature": compute_video_signature(video_path),
        "captions": captions,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file_for(video_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def query_cache_state(video_id: str, video_path: str) -> dict:
    f = cache_file_for(video_id)
    if not f.exists():
        return {"state": "absent"}

    try:
        payload = json.load(open(f, "r", encoding="utf-8"))
    except Exception:
        return {"state": "absent", "reason": "cache_unreadable"}

    if payload.get("video_id") != video_id:
        return {
            "state": "wrong_video_id",
            "reason": "cache video_id field does not match request",
        }

    sig = compute_video_signature(video_path)
    if payload.get("video_signature") != sig:
        return {
            "state": "signature_mismatch",
            "reason": "video file changed since cache was written, or cache contains captions from a different video",
        }

    age = _hours_since(payload.get("written_at", ""))
    fresh = age < STALE_THRESHOLD_HOURS

    return {
        "state": "valid",
        "fresh": fresh,
        "age_hours": round(age, 3),
        "captions": payload.get("captions", []),
        "written_at": payload.get("written_at"),
    }
