from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from config import LOGS_DIR


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_audit_entry(state: dict, node: str, row: dict, transition_idx: int) -> dict:
    entry = {
        "idx": transition_idx,
        "ts": _ts(),
        "node": node,
        "payload": row,
    }
    audit = state.setdefault("audit", [])
    audit.append(entry)

    run_id = state.get("run_id") or "anonymous"
    path = LOGS_DIR / f"run_{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def finalize_audit_path(state: dict) -> Path:
    run_id = state.get("run_id") or "anonymous"
    return LOGS_DIR / f"run_{run_id}.jsonl"


def make_run_id() -> str:
    return f"{int(time.time())}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
