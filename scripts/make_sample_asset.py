"""Generate the synthetic sample video used by the demo.

Produces a 20-second MP4 with a solid blue background and a continuous
synthesized music bed (A-major triad). No burned-in text, no TTS, no
speech — just raw visual + audio material. Subtitles and music ducking
are applied later by the agent's export step or by make_conflict_demo.py.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from config import FFMPEG_EXE, ASSETS_DIR


def make_sample_asset(path: Path | None = None, duration: float = 20.0) -> Path:
    if path is None:
        path = ASSETS_DIR / "sample.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Blue background video
    video_src = f"color=c=0x1f3a5f:s=1280x720:d={duration}:r=30"

    # A-major triad music bed: A2 (110 Hz) + C#3 (138.59 Hz) + E3 (164.81 Hz)
    # amplitude 0.15 each — gentle background music
    audio_src = (
        f"aevalsrc="
        f"0.15*sin(2*PI*110*t)+"
        f"0.15*sin(2*PI*138.59*t)+"
        f"0.15*sin(2*PI*164.81*t):"
        f"d={duration}:s=44100"
    )

    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", video_src,
        "-f", "lavfi", "-i", audio_src,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-1500:]}")
    print(f"sample asset written: {path} ({path.stat().st_size} bytes)")
    return path


if __name__ == "__main__":
    p = make_sample_asset()
    print(p)