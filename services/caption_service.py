from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = ROOT / "scripts" / "scenarios.json"

SCENARIO_NAME = os.environ.get("SCENARIO", "conflict")
SCENARIOS = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8-sig"))
SCENARIO = SCENARIOS.get(SCENARIO_NAME)
if SCENARIO is None:
    SCENARIO = SCENARIOS["conflict"]

app = FastAPI(title="Caption Service")


@app.get("/timings")
def timings(video_id: str):
    if SCENARIO.get("caption_service") == "offline":
        raise HTTPException(status_code=503, detail="caption_service simulated offline for this scenario")
    captions = SCENARIO.get("captions", [])
    return {"video_id": video_id, "source": "caption_service", "captions": captions}


@app.get("/health")
def health():
    return {"status": "online", "scenario": SCENARIO_NAME}
