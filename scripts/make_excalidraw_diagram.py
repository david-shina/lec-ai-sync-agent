"""Generate docs/architecture.excalidraw - a visual whiteboard of the full agent system.

Run:
    python scripts/make_excalidraw_diagram.py
Output:
    docs/architecture.excalidraw  (import via excalidraw.com -> File -> Open)
"""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "architecture.excalidraw"

_elements: list[dict] = []
_nonce_rng = random.Random(42)
_seed_rng = random.Random(7)


def _id() -> str:
    return uuid.uuid4().hex


def _seed() -> int:
    return _seed_rng.randint(1000, 2**31 - 1)


def _nonce() -> int:
    return _nonce_rng.randint(1000, 2**31 - 1)


def _common(t, x, y, w, h, stroke="#1e1e1e", bg="transparent",
            fillStyle="hachure", strokeWidth=1, strokeStyle="solid",
            roundness=None):
    return {
        "id": _id(), "type": t, "x": x, "y": y, "width": w, "height": h,
        "strokeColor": stroke, "backgroundColor": bg, "fillStyle": fillStyle,
        "strokeWidth": strokeWidth, "strokeStyle": strokeStyle,
        "roughness": 1, "opacity": 100, "angle": 0, "groupIds": [],
        "frameId": None, "roundness": roundness, "seed": _seed(),
        "version": 1, "versionNonce": _nonce(), "isDeleted": False,
        "boundElements": [], "updated": 1672531200000,
        "link": None, "locked": False,
    }


def rect(x, y, w, h, stroke="#1e1e1e", bg="transparent",
         fillStyle="hachure", strokeWidth=1, roundness_type=None):
    rd = ({"type": roundness_type} if roundness_type is not None else None)
    el = _common("rectangle", x, y, w, h, stroke, bg, fillStyle,
                 strokeWidth=strokeWidth, roundness=rd)
    _elements.append(el)
    return el


def text(x, y, t, fontSize=16, color="#1e1e1e", w=200, h=24,
         align="left"):
    el = _common("text", x, y, w, h, color, strokeWidth=1)
    el.update({
        "fontSize": fontSize, "fontFamily": 1, "text": t,
        "textAlign": align, "verticalAlign": "top",
        "containerId": None, "originalText": t,
        "baseline": int(fontSize * 0.85), "lineHeight": 1.25,
    })
    _elements.append(el)
    return el


def arrow(x1, y1, x2, y2, color="#495057", strokeWidth=2, dashed=False):
    el = _common("arrow", x1, y1, max(1, abs(x2 - x1)), max(1, abs(y2 - y1)),
                 stroke=color, strokeWidth=strokeWidth,
                 strokeStyle="dashed" if dashed else "solid",
                 roundness={"type": 2})
    el.update({
        "points": [[0.0, 0.0], [float(x2 - x1), float(y2 - y1)]],
        "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
    })
    _elements.append(el)
    return el


BLUE_STROKE = "#1971c2"
BLUE_BG = "#a5d8ff"
GREEN_STROKE = "#2f9e44"
GREEN_BG = "#b2f2bb"
ORANGE_STROKE = "#e8590c"
ORANGE_BG = "#ffe8cc"
GREY_STROKE = "#868e96"
GREY_BG = "#e9ecef"
ARROW_COLOR = "#495057"


text(560, 30, "Video Export Conflict-Reconciliation Agent",
     fontSize=32, color="#1e1e1e", w=700, h=44)

ZONE_A_Y = 130
BOX_W, BOX_H = 220, 80

rect(80, ZONE_A_Y, BOX_W, BOX_H, BLUE_STROKE, BLUE_BG, "hachure", 2, 3)
text(105, ZONE_A_Y + 12, "Caption Service", 18, w=200, h=24)
text(105, ZONE_A_Y + 38, ":8001 /timings", 12, "#495057", w=200, h=18)
text(105, ZONE_A_Y + 58, "online/off", 12, "#495057", w=200, h=18)

rect(340, ZONE_A_Y, BOX_W, BOX_H, BLUE_STROKE, BLUE_BG, "hachure", 2, 3)
text(365, ZONE_A_Y + 12, "Audio Mixer", 18, w=200, h=24)
text(365, ZONE_A_Y + 38, ":8002 /timings", 12, "#495057", w=200, h=18)
text(365, ZONE_A_Y + 58, "online/off", 12, "#495057", w=200, h=18)

rect(600, ZONE_A_Y, BOX_W, BOX_H, BLUE_STROKE, BLUE_BG, "hachure", 2, 3)
text(625, ZONE_A_Y + 12, "Video File", 18, w=200, h=24)
text(625, ZONE_A_Y + 38, "sample.mp4", 12, "#495057", w=200, h=18)
text(625, ZONE_A_Y + 58, "ffprobe", 12, "#495057", w=200, h=18)

rect(860, ZONE_A_Y, BOX_W, BOX_H, BLUE_STROKE, BLUE_BG, "hachure", 2, 3)
text(885, ZONE_A_Y + 12, "Cache Directory", 18, w=200, h=24)
text(885, ZONE_A_Y + 38, ".cache/captions_*", 12, "#495057", w=200, h=18)
text(885, ZONE_A_Y + 58, "SHA256 signature", 12, "#495057", w=200, h=18)

AGENT_X, AGENT_Y, AGENT_W, AGENT_H = 60, 290, 1080, 540

rect(AGENT_X, AGENT_Y, AGENT_W, AGENT_H, GREEN_STROKE, "transparent",
     "solid", 3)
text(AGENT_X + 380, AGENT_Y + 12, "LangGraph Agent", 24, GREEN_STROKE,
     w=300, h=32)

rect(200, AGENT_Y + 70, 220, 80, GREEN_STROKE, GREEN_BG, "hachure", 2, 3)
text(220, AGENT_Y + 82, "Planner (LLM)", 18, w=200, h=24)
text(220, AGENT_Y + 110, "structured output", 12, "#495057", w=200, h=18)
text(220, AGENT_Y + 128, "DEFAULT_PLAN fallback", 12, "#495057", w=200, h=18)

rect(200, AGENT_Y + 180, 240, 80, GREEN_STROKE, GREEN_BG, "hachure", 2, 3)
text(220, AGENT_Y + 192, "Step Executor (code)", 18, w=220, h=24)
text(220, AGENT_Y + 220, "dispatch to tools", 12, "#495057", w=220, h=18)
text(220, AGENT_Y + 238, "loops plan[step_idx]", 12, "#495057", w=220, h=18)

rect(560, AGENT_Y + 180, 240, 80, GREEN_STROKE, GREEN_BG, "hachure", 2, 3)
text(580, AGENT_Y + 192, "Decider (LLM + tool)", 18, w=220, h=24)
text(580, AGENT_Y + 220, "evaluate_option_costs", 12, "#495057", w=220, h=18)
text(580, AGENT_Y + 238, "Groq fallback if >=2 conflicts", 12, "#495057", w=220, h=18)

rect(820, AGENT_Y + 180, 240, 80, GREEN_STROKE, GREEN_BG, "hachure", 2, 3)
text(840, AGENT_Y + 192, "Replanner (LLM)", 18, w=220, h=24)
text(840, AGENT_Y + 220, "source_offline branch", 12, "#495057", w=220, h=18)
text(840, AGENT_Y + 238, "unresolvable_tie branch", 12, "#495057", w=220, h=18)

rect(560, AGENT_Y + 320, 240, 80, GREEN_STROKE, GREEN_BG, "hachure", 2, 3)
text(585, AGENT_Y + 332, "Exporter", 18, w=200, h=24)
text(585, AGENT_Y + 360, "ffmpeg subprocess", 12, "#495057", w=200, h=18)
text(585, AGENT_Y + 378, "vf=subtitles, af=volume", 12, "#495057", w=200, h=18)

rect(820, AGENT_Y + 320, 240, 80, GREEN_STROKE, GREEN_BG, "hachure", 2, 3)
text(850, AGENT_Y + 332, "Audit Writer", 18, w=200, h=24)
text(850, AGENT_Y + 360, "logs/run_<ts>.jsonl", 12, "#495057", w=200, h=18)
text(850, AGENT_Y + 378, "one row per transition", 12, "#495057", w=200, h=18)

DETC_Y = 880
DETC_BOX_W, DETC_BOX_H = 200, 80

detc_labels = [
    ("Cost Model", "drift = start + 0.5*end", "R = 8.0"),
    ("Detector", "pairing heuristic", "EPS_START=0.15"),
    ("Tiebreaker", "min_rerender", "then trust_audio"),
    ("Cache Integrity", "SHA256 size+64KBx2", "freshness < 1h"),
    ("Validation", "overlays LLM output", "code wins, corrected=true"),
]
for i, (title, sub1, sub2) in enumerate(detc_labels):
    x = 80 + i * 230
    rect(x, DETC_Y, DETC_BOX_W, DETC_BOX_H, ORANGE_STROKE, ORANGE_BG,
         "hachure", 2, 3)
    text(x + 20, DETC_Y + 10, title, 16, w=180, h=22)
    text(x + 20, DETC_Y + 34, sub1, 12, "#495057", w=180, h=18)
    text(x + 20, DETC_Y + 52, sub2, 12, "#495057", w=180, h=18)

OUT_X = 1180
out_labels = [
    ("out.mp4", "happy path", False),
    ("out_needs_review.mp4", "Failure B (tie)", False),
    ("out_degraded.mp4", "Failure A, cache ok", False),
    ("out_no_captions.mp4", "Failure A, no cache", False),
    ("logs/run_<ts>.jsonl", "audit trail", True),
]
for i, (title, sub, is_log) in enumerate(out_labels):
    y_bp = 290 + i * 120
    rect(OUT_X, y_bp, 220, 80, GREY_STROKE, GREY_BG,
         "hachure" if is_log else "cross-hatch", 2, 3)
    text(OUT_X + 20, y_bp + 12, title, 14, w=200, h=22)
    text(OUT_X + 20, y_bp + 36, sub, 12, "#495057", w=200, h=18)
    text(OUT_X + 20, y_bp + 56, "audit" if is_log else "output", 12, "#495057", w=200, h=18)


def arrow_with_label(x1, y1, x2, y2, label, label_offset_x=0,
                     label_offset_y=-20, color=ARROW_COLOR, dashed=False):
    arrow(x1, y1, x2, y2, color=color, strokeWidth=2, dashed=dashed)
    text(min(x1, x2) + label_offset_x, min(y1, y2) + label_offset_y,
         label, 11, "#495057", w=200, h=16, align="left")


arrow_with_label(190, ZONE_A_Y + 80, 290, AGENT_Y + 180, "captions[] (httpx 3s)")
arrow_with_label(450, ZONE_A_Y + 80, 320, AGENT_Y + 180, "ducks[] (httpx 3s)")
arrow_with_label(710, ZONE_A_Y + 80, 350, AGENT_Y + 180, "metadata{segments}")
arrow_with_label(970, ZONE_A_Y + 80, 1130, 920, "cache_state (if offline)",
                 color="#fa5252", dashed=True)
arrow_with_label(310, AGENT_Y + 150, 310, AGENT_Y + 180, "plan[]")
arrow_with_label(440, AGENT_Y + 220, 560, AGENT_Y + 220, "if conflicts != []")
arrow_with_label(440, AGENT_Y + 200, 820, AGENT_Y + 200, "if SourceOffline",
                 color="#fa5252", dashed=True)
arrow_with_label(800, AGENT_Y + 230, 820, AGENT_Y + 260, "if computed_tie",
                 color="#fa5252", dashed=True)
arrow_with_label(560, AGENT_Y + 240, 440, AGENT_Y + 240, "verdict")
arrow_with_label(820, AGENT_Y + 230, 440, AGENT_Y + 260, "new plan (no retry)",
                 color="#fa5252")
arrow_with_label(940, AGENT_Y + 260, 680, AGENT_Y + 320, "tiebreaker chosen",
                 color="#fa5252")
arrow_with_label(320, AGENT_Y + 260, 560, AGENT_Y + 360, "if all steps done")
arrow_with_label(800, AGENT_Y + 360, 820, AGENT_Y + 360, "transitions[]")
arrow_with_label(800, AGENT_Y + 380, 1200, 380, "out{suffix}.mp4")
arrow_with_label(1060, AGENT_Y + 380, 1280, 770, "JSONL append")
arrow_with_label(180, 880, 560, AGENT_Y + 230, "CostReport to decider",
                 label_offset_y=-200)
arrow_with_label(410, 880, 820, AGENT_Y + 240, "cache_state to replanner",
                 label_offset_y=-200, color="#fa5252", dashed=True)
arrow_with_label(640, 880, 870, AGENT_Y + 290, "tiebreaker applied",
                 label_offset_y=-220, color="#fa5252")
arrow_with_label(870, 880, 580, AGENT_Y + 290, "validate verdict + replan",
                 label_offset_y=-280)


scene = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": _elements,
    "appState": {
        "viewBackgroundColor": "#ffffff",
        "currentItemFontFamily": 1,
        "gridSize": None,
    },
    "files": {},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(scene, indent=2), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(_elements)} elements)")