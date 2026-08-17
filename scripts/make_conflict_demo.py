"""Produce the conflict demo video.

Reads the 'conflict' scenario timings from scenarios.json, then renders
sample.mp4 with:
  - Subtitles burned at CAPTION-SERVICE times (yellow, top)
  - Music ducking applied at AUDIO-MIXER times (the disagreeing times)

Result: the viewer sees subtitles appear at one moment and hears the
music dip at a different moment — every conflict is visible and audible
in a single 20-second playthrough.

Usage:
    python scripts/make_conflict_demo.py
    python scripts/make_conflict_demo.py --scenario conflict
    python scripts/make_conflict_demo.py --input-video path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import FFMPEG_EXE, ASSETS_DIR, SCENARIOS_PATH


def _write_srt(captions: list, path: str) -> None:
    def _fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = t - h * 3600 - m * 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    lines = []
    for idx, c in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt(c['start'])} --> {_fmt(c['end'])}")
        lines.append(str(c["text"]))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _ffmpeg_escape_path(p: str) -> str:
    norm = p.replace("\\", "/")
    return norm.replace(":", "\\\\:")


def make_conflict_demo(scenario_name: str = "conflict",
                       input_video: str | None = None,
                       output_path: str | None = None) -> Path:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8-sig"))
    scenario = scenarios.get(scenario_name)
    if scenario is None:
        raise ValueError(f"unknown scenario: {scenario_name}")

    if input_video is None:
        input_video = str(ASSETS_DIR / "sample.mp4")
    if output_path is None:
        output_path = str(ASSETS_DIR / "conflict_demo.mp4")

    captions = scenario["captions"]
    ducks = scenario["ducks"]

    # Write SRT at CAPTION-SERVICE times (the raw, unaligned times)
    fd, srt_path = tempfile.mkstemp(prefix="conflict_sub_", suffix=".srt")
    os.close(fd)
    _write_srt(captions, srt_path)
    escaped_srt = _ffmpeg_escape_path(srt_path)

    # Build audio filter: volume ducking at AUDIO-MIXER times (raw, unaligned)
    audio_parts = []
    for d in ducks:
        st = float(d["start"]); en = float(d["end"])
        gain = float(d.get("gain_reduction_db", -12.0))
        ratio = max(0.0, min(1.0, 10 ** (gain / 20.0)))
        audio_parts.append(
            f"volume=enable='between(t,{st},{en})':volume={ratio}"
        )
    af = ",".join(audio_parts) if audio_parts else "anull"

    vf = f"subtitles={escaped_srt}"

    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_video,
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        os.unlink(srt_path)
    except Exception:
        pass

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-1500:]}")

    out = Path(output_path)
    print(f"conflict demo written: {out} ({out.stat().st_size} bytes)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="conflict")
    parser.add_argument("--input-video", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    make_conflict_demo(args.scenario, args.input_video, args.output)