from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
from langchain_core.tools import tool

from agent.state import (
    Cue, Duck, Conflict, VideoMetadata, Segment, CostReport,
)
from agent.cost_model import evaluate_option_costs as _evaluate_option_costs
from agent.cache_integrity import (
    query_cache_state as _query_cache_state,
    write_cache as _write_cache,
)
from agent.metadata import query_video_metadata as _query_video_metadata
from agent.detector import detect_conflicts as _detect_conflicts
from agent.tiebreaker import apply_tiebreaker as _apply_tiebreaker
from config import HTTP_TIMEOUT_SECONDS, FFMPEG_EXE, ASSETS_DIR


class SourceOffline(Exception):
    def __init__(self, source: str, detail: str):
        super().__init__(f"{source} offline: {detail}")
        self.source = source
        self.detail = detail


CAPTION_SERVICE_URL = "http://localhost:8001/timings"
AUDIO_SERVICE_URL = "http://localhost:8002/timings"


def fetch_caption_source(video_id: str, video_path: str | None = None) -> dict:
    try:
        r = httpx.get(CAPTION_SERVICE_URL, params={"video_id": video_id},
                      timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout,
            httpx.RemoteProtocolError) as e:
        raise SourceOffline("caption_service", str(e))
    payload = r.json()
    captions = payload.get("captions", [])
    if video_path:
        try:
            _write_cache(video_id, video_path, captions)
        except Exception:
            pass
    return {"captions": captions}


def fetch_audio_source(video_id: str) -> dict:
    try:
        r = httpx.get(AUDIO_SERVICE_URL, params={"video_id": video_id},
                      timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout,
            httpx.RemoteProtocolError) as e:
        raise SourceOffline("audio_mixer", str(e))
    payload = r.json()
    return {"ducks": payload.get("ducks", [])}


def query_video_metadata_tool(video_path: str, segment_overrides: dict | None = None) -> dict:
    md = _query_video_metadata(video_path, segment_overrides)
    return md.model_dump()


def query_metadata_cache(video_id: str, video_path: str | None = None) -> dict:
    cs = _query_cache_state(video_id, video_path or "")
    if cs.get("state") != "valid":
        raise SourceOffline("caption_cache", f"cache_state={cs.get('state')}")
    return {"captions": cs.get("captions", [])}


def query_cache_state_tool(video_id: str, video_path: str) -> dict:
    return _query_cache_state(video_id, video_path)


def detect_conflicts_tool(captions: list, ducks: list) -> dict:
    conflicts = _detect_conflicts(captions, ducks)
    return {"conflicts": [c.model_dump() for c in conflicts]}


def apply_tiebreaker_tool(cost_report: dict) -> dict:
    cr = CostReport(**cost_report)
    return _apply_tiebreaker(cr)


@tool
def evaluate_option_costs(conflict: dict, metadata: dict) -> str:
    """Return a JSON CostReport for one conflict given video metadata dict."""
    c = Conflict(**conflict) if not isinstance(conflict, Conflict) else conflict
    m = VideoMetadata(**metadata) if not isinstance(metadata, VideoMetadata) else metadata
    cr = _evaluate_option_costs(c, m)
    return cr.model_dump_json(indent=2)


DECIDER_TOOLS = [evaluate_option_costs]


def build_resolved_timeline(captions: list, ducks: list,
                            chosen_by_conflict: dict | None,
                            no_captions: bool = False) -> dict:
    cap_by_id = {c["id"]: dict(c) for c in (captions or [])}
    duck_by_id = {d["id"]: dict(d) for d in (ducks or [])}

    for cid, option in (chosen_by_conflict or {}).items():
        if option is None or "/" not in cid:
            continue
        cap_id, duck_id = cid.split("/", 1)
        cap = cap_by_id.get(cap_id)
        duck = duck_by_id.get(duck_id)
        if cap is None or duck is None:
            continue
        if option == "trust_caption":
            duck["start"] = cap["start"]
            duck["end"] = cap["end"]
        elif option == "trust_audio":
            cap["start"] = duck["start"]
            cap["end"] = duck["end"]
        elif option == "re_render_duck":
            duck["start"] = cap["start"]
            duck["end"] = cap["end"]

    return {
        "captions": list(cap_by_id.values()),
        "ducks": list(duck_by_id.values()),
        "no_captions": no_captions,
    }


def _ffmpeg_escape_path(p: str) -> str:
    norm = p.replace("\\", "/")
    return norm.replace(":", r"\\:")


def _space_free_temp_srt(captions: list) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(prefix="sub_", suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        _write_srt(captions, f)
    return path


def export_video(resolved_timeline: dict, output_path: str,
                 input_video: str | None = None) -> dict:
    if input_video is None:
        input_video = str(ASSETS_DIR / "sample.mp4")

    out_dir = Path(output_path).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    captions = resolved_timeline.get("captions", [])
    ducks = resolved_timeline.get("ducks", [])
    no_captions = resolved_timeline.get("no_captions", False)

    tmp_srt_path: str | None = None
    video_chain = []
    if captions and not no_captions:
        try:
            tmp_srt_path = _space_free_temp_srt(captions)
            escaped = _ffmpeg_escape_path(tmp_srt_path)
            video_chain.append(f"subtitles={escaped}")
        except Exception as e:
            tmp_srt_path = None
            return {"output_path":output_path,"ffmpeg_cmd":[],
                    "returncode":-1,"stderr_tail":f"srt build failed: {e}",
                    "success":False,"subtitle_burn_failed":True}

    audio_chain = []
    for d in ducks:
        st = float(d["start"]); en = float(d["end"])
        gain = float(d.get("gain_reduction_db", -12.0))
        ratio = max(0.0, min(1.0, 10 ** (gain / 20.0)))
        audio_chain.append(
            f"volume=enable='between(t,{st},{en})':volume={ratio}"
        )

    vf = ",".join(video_chain) if video_chain else "null"
    af = ",".join(audio_chain) if audio_chain else "anull"

    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_video,
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if tmp_srt_path:
        try:
            os.unlink(tmp_srt_path)
        except Exception:
            pass

    return {
        "output_path": str(out_dir),
        "ffmpeg_cmd": cmd,
        "returncode": proc.returncode,
        "stderr_tail": (proc.stderr or "")[-400:],
        "success": proc.returncode == 0,
    }


def _write_srt(captions: list, target) -> None:
    def _fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = t - h * 3600 - m * 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    lines = []
    for idx, c in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt(c['start'])} --> {_fmt(c['end'])}")
        lines.append(str(c["text"]))
        lines.append("")
    body = "\n".join(lines)

    if hasattr(target, "write"):
        target.write(body)
    else:
        Path(target).write_text(body, encoding="utf-8")


def dispatch_tool(tool_name: str, state: dict) -> dict:
    video_id = state.get("video_id", "")
    video_path = state.get("video_path", "")

    if tool_name == "fetch_caption_source":
        return fetch_caption_source(video_id, video_path)
    if tool_name == "fetch_audio_source":
        return fetch_audio_source(video_id)
    if tool_name == "query_video_metadata":
        return query_video_metadata_tool(video_path, state.get("segment_overrides"))
    if tool_name == "query_metadata_cache":
        return query_metadata_cache(video_id, video_path)
    if tool_name == "query_cache_state":
        return query_cache_state_tool(video_id, video_path)
    if tool_name == "detect_conflicts":
        return detect_conflicts_tool(state.get("captions", []), state.get("ducks", []))
    if tool_name == "evaluate_option_costs":
        c = (state.get("conflicts") or [{}])[0]
        m = state.get("metadata") or {}
        return {"cost_report": json.loads(evaluate_option_costs.invoke(
            {"conflict": c, "metadata": m}))}
    if tool_name == "apply_tiebreaker":
        # If we have multiple cost reports, use multi-tiebreaker
        cost_reports = state.get("cost_reports_by_conflict") or {}
        if cost_reports and len(cost_reports) > 1:
            result = apply_tiebreaker_tool(cost_reports_by_conflict=cost_reports)
            # Update chosen_options_by_conflict with all winners
            existing = dict(state.get("chosen_options_by_conflict") or {})
            for cid, chosen in result.get("chosen_by_conflict", {}).items():
                existing[cid] = chosen
            state["chosen_options_by_conflict"] = existing
            return result
        else:
            # Single conflict case
            result = apply_tiebreaker_tool(cost_report=state.get("cost_report") or {})
            cid = result.get("conflict_id")
            chosen = result.get("chosen")
            if cid and chosen:
                existing = dict(state.get("chosen_options_by_conflict") or {})
                existing[cid] = chosen
                state["chosen_options_by_conflict"] = existing
            return result
    if tool_name == "export_video":
        captions = state.get("captions") or []
        ducks = state.get("ducks") or []
        chosen = state.get("chosen_options_by_conflict") or {}
        no_captions = state.get("no_captions", False)
        tl = build_resolved_timeline(captions, ducks, chosen, no_captions)
        state["resolved_timeline"] = tl
        suffix = state.get("output_suffix", "")
        out = str(ASSETS_DIR / f"out{suffix}.mp4")
        return export_video(tl, out, video_path)

    raise ValueError(f"unknown tool in plan: {tool_name}")
import os
