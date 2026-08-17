"""Run a single demo scenario end-to-end: services + cache + agent + audit print.

Usage:
    python scripts/run_demo.py --scenario conflict
    python scripts/run_demo.py --scenario unresolvable_tie
    python scripts/run_demo.py --scenario source_offline_cache_fresh
    python scripts/run_demo.py --scenario source_offline_cache_empty
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.audit import make_run_id
from agent.cache_integrity import (
    write_cache, cache_file_for, query_cache_state,
)
from agent.graph import build_graph

from config import ASSETS_DIR, SCENARIOS_PATH


def setup_progress_logging(run_id: str) -> logging.Logger:
    """Configure logging for demo progress - console + file."""
    log = logging.getLogger("demo.progress")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    fmt = logging.Formatter("[%(name)s] %(message)s")

    # Console (stdout)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"demo_progress_{run_id}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    # Suppress other loggers' noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)

    return log


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="conflict",
                   choices=["conflict", "unresolvable_tie",
                            "source_offline_cache_fresh",
                            "source_offline_cache_stale",
                            "source_offline_cache_empty",
                            "source_offline_cache_signature_mismatch"])
    p.add_argument("--print-log", action="store_true",
                   help="Print a human-readable run trace (plans, replans, routing, costs) to stdout")
    p.add_argument("--input-video", default=None,
                   help="Path to an alternative input video (default: assets/sample.mp4)")
    p.add_argument("--keep-services", action="store_true",
                   help="Do not kill the FastAPI services on exit (useful while iterating)")
    return p.parse_args()


def load_scenarios() -> dict:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8-sig"))


def start_services(scenario: dict, scenario_name: str) -> list:
    env = {**os.environ, "SCENARIO": scenario_name}
    procs: list[tuple[str, subprocess.Popen]] = []

    if scenario.get("caption_service") == "online":
        p = subprocess.Popen(
            ["python", "-m", "uvicorn", "services.caption_service:app", "--port", "8001"],
            cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("caption", p))
    if scenario.get("audio_service") == "online":
        p = subprocess.Popen(
            ["python", "-m", "uvicorn", "services.audio_service:app", "--port", "8002"],
            cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("audio", p))

    time.sleep(2.5)
    return procs


def stop_services(procs: list) -> None:
    for _, p in procs:
        try:
            p.terminate()
            p.wait(timeout=4)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def setup_cache(scenario: dict, video_id: str, video_path: str, captions: list) -> None:
    cache_path = cache_file_for(video_id)

    if not scenario.get("seed_cache"):
        if cache_path.exists():
            cache_path.unlink()
        return

    write_cache(video_id, video_path, captions)

    freshness = scenario.get("cache_freshness", "fresh")
    if freshness == "stale":
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
        payload["written_at"] = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    corruption = scenario.get("corrupt_cache")
    if corruption == "signature":
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
        payload["video_signature"] = "0" * 64
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif corruption == "wrong_video_id":
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
        payload["video_id"] = "different_video"
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_sample_asset(video_path: str) -> None:
    if Path(video_path).exists():
        return
    from scripts.make_sample_asset import make_sample_asset
    make_sample_asset()


def print_summary(state: dict) -> None:
    print()
    print("=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"  Scenario:        {state.get('scenario')}")
    print(f"  Run ID:          {state.get('run_id')}")
    print(f"  Output file:     {state.get('output_path') or '(none)'}")
    print(f"  Export success:  {state.get('export_success')}")
    print(f"  Degraded mode:   {state.get('degraded_mode', False)}")
    print(f"  Needs review:    {state.get('needs_review', False)}")
    print(f"  Conflict count:  {len(state.get('conflicts') or [])}")
    print(f"  Verdicts:        {len(state.get('verdicts') or [])}")
    if state.get("verdicts"):
        for v in state["verdicts"]:
            print(f"    - {v.get('conflict_id')}: chosen={v.get('chosen_option')!r} tie={v.get('tie')} conf={v.get('confidence')} source={v.get('source')}")
    if state.get("chosen_options_by_conflict"):
        print(f"  Chosen per conflict: {state['chosen_options_by_conflict']}")
    print(f"  Audit rows:      {len(state.get('audit') or [])}")
    run_id = state.get('run_id')
    print(f"  Audit log path:  {ROOT / 'logs' / f'run_{run_id}.jsonl'}")
    print("=" * 70)


def _brief_result(result: dict) -> str:
    if not result:
        return "done"
    if "captions" in result:
        return f"{len(result['captions'])} captions fetched"
    if "ducks" in result:
        return f"{len(result['ducks'])} ducks fetched"
    if "duration" in result:
        return f"{round(result['duration'], 1)}s, {len(result.get('segments', []))} segments"
    if "conflicts" in result:
        return f"{len(result['conflicts'])} conflicts detected"
    if "output_path" in result:
        ok = result.get("success", False)
        return ("OK: " if ok else "FAILED: ") + str(result.get("output_path", ""))
    if "chosen" in result:
        return f"chose {result.get('chosen')}"
    if "state" in result:
        return f"cache_state={result.get('state')}"
    return "done"


def _format_steps(steps) -> list[str]:
    return [f"    {i}. {s.get('tool')} - {s.get('intent', '')}".rstrip()
            for i, s in enumerate(steps, start=1)]


def _cost_line(options: list) -> str:
    return " | ".join(f"{o.get('option')}={o.get('cost')}" for o in options)


def print_trace(state: dict) -> None:
    print()
    print("#" * 70)
    print("# RUN TRACE")
    print("#" * 70)
    for row in state.get("audit", []):
        node = row.get("node")
        payload = row.get("payload") or {}
        idx = row.get("idx")

        if payload.get("event") == "route":
            print(f"  {payload.get('from')} -> {payload.get('to')}   ({payload.get('reason')})")
            continue

        if node == "planner":
            if payload.get("no_op_before"):
                continue
            if payload.get("plan"):
                print(f"[{idx}] planner (source={payload.get('source')})")
                print(f"    PLAN ({len(payload['plan'])} steps):")
                for line in _format_steps(payload["plan"]):
                    print(line)
                if payload.get("rationale"):
                    print(f"    rationale: {payload.get('rationale')}")
            else:
                print(f"[{idx}] planner FAILED: {payload.get('reject_reason')}")
            continue

        if node == "step_executor":
            if payload.get("error"):
                print(f"[{idx}] step_executor (step {payload.get('step')}) {payload.get('tool')} "
                      f"-> ERROR: {payload.get('error')} ({payload.get('source')})")
                continue
            tool = payload.get("tool")
            if tool:
                n = payload.get("plan_len") or len(state.get("plan") or [])
                print(f"[{idx}] step_executor ({payload.get('step') + 1}/{n}) {tool} "
                      f"-> {_brief_result(payload.get('result') or {})}")
                if tool == "apply_tiebreaker" and payload.get("result", {}).get("both_options"):
                    tied = _cost_line(payload["result"]["both_options"])
                    print(f"      tied costs: {tied}   (policy={payload['result'].get('policy')})")
            continue

        if node == "decider":
            if payload.get("pre_computed_cost_reports"):
                print(f"[{idx}] decider: cost reports computed for "
                      f"{len(payload['pre_computed_cost_reports'])} conflict(s)")
                continue
            if payload.get("source") == "deterministic_fallback" and payload.get("error"):
                print(f"[{idx}] decider: deterministic fallback ({payload.get('error')})")
                continue
            vv = payload.get("validated_verdicts")
            if vv:
                crs = state.get("cost_reports_by_conflict") or {}
                print(f"[{idx}] decider" + (f" (src={payload.get('source')})"
                                            if payload.get("source") else ""))
                for v in vv:
                    cid = v.get("conflict_id")
                    cr = crs.get(cid) or {}
                    opts = cr.get("options") or []
                    print(f"    {cid} -> {v.get('chosen_option')}  "
                          f"(tie={v.get('tie')}, conf={v.get('confidence')})")
                    if opts:
                        print(f"      costs: {_cost_line(opts)}   "
                              f"(seg={cr.get('segment_type')}, margin={cr.get('marginal_advantage')})")
            continue

        if node == "replanner":
            if payload.get("trigger_failure_mode"):
                print(f"[{idx}] replanner triggered by failure_mode={payload['trigger_failure_mode']}")
            elif payload.get("plan"):
                print(f"[{idx}] replanner REPLAN (strategy={payload.get('strategy')}, "
                      f"source={payload.get('source')}):")
                for line in _format_steps(payload["plan"]):
                    print(line)
                if payload.get("rationale"):
                    print(f"    rationale: {payload.get('rationale')}")
            elif payload.get("error"):
                print(f"[{idx}] replanner: {payload.get('error')} (strategy={payload.get('strategy')})")
            continue

        if node == "audit_writer":
            summary = payload.get("summary") or {}
            if summary.get("planner_failed"):
                print(f"[{idx}] audit_writer: RUN ABORTED - planner failed ({summary.get('reason')})")
            else:
                print(f"[{idx}] audit_writer: output={payload.get('output_path')} "
                      f"export_success={payload.get('export_success')} "
                      f"needs_review={summary.get('needs_review')}")
            continue

        print(f"[{idx}] {node}: {json.dumps(payload, default=str)}")
    print()


def print_log(state: dict) -> None:
    print_trace(state)


def main() -> int:
    args = parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    scenario_name = args.scenario
    scenarios = load_scenarios()
    if scenario_name not in scenarios:
        raise SystemExit(f"Unknown scenario: {scenario_name!r}. Available: {list(scenarios)}")
    scenario = scenarios.get(scenario_name)

    video_id = "sample"
    if args.input_video:
        video_path = args.input_video
    else:
        video_path = str(ASSETS_DIR / "sample.mp4")

    ensure_sample_asset(video_path)
    setup_cache(scenario, video_id, video_path, scenario.get("captions", []))

    procs: list = []
    if scenario.get("caption_service") == "online" or scenario.get("audio_service") == "online":
        procs = start_services(scenario, scenario_name)

    procs_to_kill = [] if args.keep_services else procs

    try:
        run_id = make_run_id()
        progress_log = setup_progress_logging(run_id)

        state = {
            "goal": f"export {video_path} reconciling caption/audio timing (scenario={scenario_name})",
            "video_id": video_id,
            "video_path": video_path,
            "scenario": scenario_name,
            "segment_overrides": scenario.get("segment_overrides"),
            "audit": [],
            "observations": [],
            "degraded_mode": False,
            "needs_review": False,
            "no_captions": False,
            "output_suffix": "",
            "step_idx": 0,
            "run_id": run_id,
            "chosen_options_by_conflict": {},
            "cache_state": None,
            "captions": None,
            "ducks": None,
            "metadata": None,
            "conflicts": None,
            "verdict": None,
            "verdicts": None,
        }

        graph = build_graph()
        progress_log.info("[INIT] Starting scenario=%s video_id=%s", scenario_name, video_id)
        final_state = graph.invoke(state)

        if final_state.get("planner_failed"):
            print()
            print("=" * 70)
            print("RUN ABORTED: planner could not produce a valid plan")
            print("=" * 70)
            print(f"  Reason: {final_state.get('planner_failure_reason')}")
            print("  No video was exported.")
            print(f"  Audit log: {ROOT / 'logs' / f'run_{final_state.get('run_id')}.jsonl'}")
            print("=" * 70)
            if args.print_log:
                print_log(final_state)
            return 1

        print_summary(final_state)
        if args.print_log:
            print_log(final_state)

        if final_state.get("export_success") is False:
            return 1
        return 0
    finally:
        stop_services(procs_to_kill)


if __name__ == "__main__":
    sys.exit(main())
