# Architecture

## LangGraph StateGraph

```
START
  │
  ▼
┌─────────────────────────┐
│  planner (LLM)          │  Reads: goal
│  structured output      │  Writes: plan
│  fallback: DEFAULT_PLAN │  Audit: {"node":"planner","plan":[...],"source":"llm|fallback"}
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│  step_executor (code)   │  Reads: plan[step_idx]
│  dispatch to @tool      │  Writes: observations[], step_idx++,
│                         │         captions|ducks|metadata|conflicts
└─────────────────────────┘
  │
  ├──[error: source_offline]──► replanner (failure_mode=source_offline)
  │                                  │ reads cache_state
  │                                  │ strategy: fallback_cache | fallback_stale_cache
  │                                  │          | export_no_captions | abort
  │                                  │ validates: canonical plan per strategy
  │                                  │ validates: no forbidden tools for this failure
  │                                  └──► step_executor (with new plan)
  │
  ├──[conflicts != [] && not resolved]──► decider
  │
  ├──[steps remain, no open conflicts]──► step_executor (loop)
  │
  └──[all steps done]──► audit_writer ──► END

┌─────────────────────────┐
│  decider (LLM + tool)   │  Tools: evaluate_option_costs
│  Groq fallback ≥2 conf  │  LLM calls tool, reads CostReport, writes Verdict
│  Deterministic fallback │
│                         │  CODE OVERLAYS (override LLM):
│                         │    1. computed_tie = margin < ε OR seg == mixed
│                         │    2. validate chosen == cost_report.best
│                         │    3. if LLM disagrees → corrected=true, force cost_model.best
└─────────────────────────┘
  │
  ├──[computed_tie == True]──► replanner (failure_mode=unresolvable_tie)
  │                               │ must contain: apply_tiebreaker + export_video
  │                               │ must NOT contain: decider, evaluate_option_costs
  │                               │ tiebreaker (code): min_rerender_then_audio_continuity
  │                               └──► step_executor (with new plan)
  │
  └──[computed_tie == False]──► step_executor (continue plan)

┌─────────────────────────┐
│  replanner (LLM)        │  Single node; prompt branches on failure_mode
│  fallback: DEFAULT_REPLAN│  Validates:
│                         │    - strategy matches cache_state or failure_mode
│                         │    - no forbidden tools for this failure mode
│                         │    - canonical plan tools enforced
└─────────────────────────┘

┌─────────────────────────┐
│  exporter (code)        │  ffmpeg: subtitles (vf) + volume ducking (af)
│                         │  Output filename from output_suffix:
│                         │    ""            → out.mp4
│                         │    "_degraded"   → out_degraded.mp4
│                         │    "_needs_review" → out_needs_review.mp4
│                         │    "_no_captions" → out_no_captions.mp4
│                         │    "_aborted"    → out_aborted.mp4
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│  audit_writer (code)    │  Writes logs/run_<timestamp>.jsonl
│                         │  One JSON row per state transition
└─────────────────────────┘
  │
  ▼
END
```

## Files and Responsibilities

| File | Role |
|------|------|
| `config.py` | Constants: weights, EPS_TIE, R, thresholds, ffmpeg path |
| `agent/state.py` | Pydantic models (Conflict, CostReport, Verdict, Plan, Replan) + AgentState TypedDict |
| `agent/cost_model.py` | `evaluate_option_costs()` — the deterministic cost formula |
| `agent/detector.py` | `detect_conflicts()` — finds caption/duck disagreements |
| `agent/tiebreaker.py` | `apply_tiebreaker()` — deterministic tiebreaker policy |
| `agent/cache_integrity.py` | `query_cache_state()` — SHA256 signature, freshness, corruption detection |
| `agent/metadata.py` | `query_video_metadata()` — ffprobe wrapper + segment classifier |
| `agent/validation.py` | `validate_plan()`, `validate_replan_against_cache_state()`, `validate_replan_against_failure_mode()`, `validate_verdict_against_cost_report()` |
| `agent/planner_defaults.py` | `DEFAULT_PLAN`, `DEFAULT_REPLAN_*`, `ALLOWED_TOOLS_BY_FAILURE`, `FORBIDDEN_TOOLS_BY_FAILURE` |
| `agent/decider_fallback.py` | `deterministic_verdicts()` — produces Verdicts from cost reports without LLM |
| `agent/tools.py` | Tool implementations: `fetch_caption_source()`, `fetch_audio_source()`, `detect_conflicts()`, `evaluate_option_costs()`, `export_video()`, `apply_tiebreaker()`, `build_resolved_timeline()` |
| `agent/prompts.py` | System prompts for planner, decider, replanner (source_offline + unresolvable_tie) |
| `agent/llm.py` | LLM bindings: Groq (llama-3.1-8b-instant) primary, LM Studio (Qwen2.5-3B) offline fallback |
| `agent/nodes.py` | LangGraph node functions + conditional routing |
| `agent/graph.py` | `build_graph()` — wires the StateGraph |
| `agent/audit.py` | JSONL audit writer |
| `services/caption_service.py` | FastAPI: GET /timings → caption cues |
| `services/audio_service.py` | FastAPI: GET /timings → duck windows |
| `scripts/scenarios.json` | 6 deterministic scenarios with timings + segment overrides |
| `scripts/make_sample_asset.py` | Generates the 20s blue-bg + music-bed sample video |
| `scripts/make_conflict_demo.py` | Generates conflict_demo.mp4 (before video) |
| `scripts/run_demo.py` | End-to-end scenario runner: starts services, runs agent, exports |
---

## Visual Whiteboard

The full-system architecture diagram is available as an ExcaliDraw scene file:

- **`docs/architecture.excalidraw`** (import via [excalidraw.com](https://excalidraw.com) → File → Open, or just drag-drop the file into the editor)

Once imported you can rearrange boxes, recolor arrows, add your own annotations, and export a PNG or SVG for the demo video or the README.

The diagram shows four color-coded zones:
- **Blue (top)** — External sources: caption service, audio mixer, video file, cache directory
- **Green (middle frame)** — LangGraph agent: planner, step executor, decider, replanner, exporter, audit writer
- **Orange (bottom row)** — Deterministic core: cost model, detector, tiebreaker, cache integrity, validation
- **Grey (right column)** — Outputs: `out.mp4`, `out_needs_review.mp4`, `out_degraded.mp4`, `out_no_captions.mp4`, `logs/run_<ts>.jsonl`

Red dashed arrows mark failure-mode edges (SourceOffline, computed_tie). Regular arrows are the happy-path control flow. Each arrow has a terse label naming the function or condition.

To regenerate the scene JSON after editing the architecture description:
```bash
python scripts/make_excalidraw_diagram.py
```
