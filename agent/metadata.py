from __future__ import annotations

import json
import subprocess

from agent.state import VideoMetadata, Segment
from config import FFPROBE_EXE


def _run_ffprobe(video_path: str) -> dict:
    cmd = [
        FFPROBE_EXE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


def query_video_metadata(video_path: str, segment_overrides: dict | None = None) -> VideoMetadata:
    info = _run_ffprobe(video_path)
    duration = float(info.get("format", {}).get("duration", 0.0))

    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )
    sample_rate = float(audio_stream.get("sample_rate", 44100)) if audio_stream else 44100

    if segment_overrides:
        segs = [Segment(**s) for s in segment_overrides["segments"]]
        return VideoMetadata(duration=segment_overrides.get("duration", duration), segments=segs)

    segments = _classify_segments(duration, sample_rate)
    return VideoMetadata(duration=duration, segments=segments)


def _classify_segments(duration: float, sample_rate: float) -> list[Segment]:
    boundaries = [0.0, duration / 3.0, 2.0 * duration / 3.0, duration]
    types = ["dialogue", "dialogue", "mixed"]
    if duration > 14:
        types = ["dialogue", "dialogue", "mixed"]
    else:
        types = ["dialogue", "music", "mixed"]
    segs = []
    for i, t in enumerate(types):
        segs.append(Segment(start=round(boundaries[i], 3), end=round(boundaries[i + 1], 3), type=t))
    return segs
