# Video Export Conflict-Reconciliation Agent

A LangGraph plan-and-execute agent that reconciles conflicts between caption
timing and audio ducking timing when two independent systems disagree, then
exports a video with the resolved (aligned) timeline.

Built for the LEC AI Engineering Intern build assessment.

---

## Quick Start

### Prerequisites

- Python 3.12+
- ffmpeg (with ffprobe) on PATH
- (Optional) `GROQ_API_KEY` in `.env` — free at [console.groq.com](https://console.groq.com/keys).
  Without it the agent falls back to deterministic mode (still works, just no LLM rationale text).
- (Optional) [LM Studio](https://lmstudio.ai/) running Qwen2.5-3B-Instruct
  at `http://localhost:1234/v1` as an offline fallback if no Groq key

### Install

```bash
pip install -r requirements.txt
cp .env.example .env  # edit if you have a Groq key or custom LM Studio URL
```

### Generate the Sample Video

```bash
python scripts/make_sample_asset.py
```

Produces `assets/sample.mp4` — a 20-second blue-background clip with a
continuous synthesized music bed (A-major triad).

### Generate the Conflict Demo Video ("Before")

```bash
python scripts/make_conflict_demo.py
```

Produces `assets/conflict_demo.mp4` — the sample video with subtitles burned
at caption-service times and music ducking at audio-mixer times (the
disagreeing times). Play this to see and hear the conflicts.

### Run the Agent

```bash
# Happy path: 2 conflicts, both resolved with trust_caption
python scripts/run_demo.py --scenario conflict

# Failure B: unresolvable tie in a mixed segment
python scripts/run_demo.py --scenario unresolvable_tie

# Failure A: caption service offline, cache fresh
python scripts/run_demo.py --scenario source_offline_cache_fresh

# Failure A: caption service offline, cache absent -> no captions
python scripts/run_demo.py --scenario source_offline_cache_empty

# Failure A: caption service offline, cache signature mismatch
python scripts/run_demo.py --scenario source_offline_cache_signature_mismatch

# Print the human-readable run trace (plans, replans, node routing, decision costs)
python scripts/run_demo.py --scenario conflict --print-log

# Use a custom input video
python scripts/run_demo.py --scenario conflict --input-video path/to/your.mp4
```

### Run Tests

```bash
python -m pytest tests/ -q
```

39 tests cover: cost model, detector, tiebreaker, replan validation, cache
integrity, and decider verdict validation.

### Start Services Manually (for debugging)

```powershell
.\scripts\start_services.ps1 -Scenario conflict
```

---

## Scenarios

| Scenario | What happens | Output file |
|----------|-------------|-------------|
| `conflict` | 2 conflicts in dialogue: duck fires 0.2s early + 0.5s late. Agent resolves both with `trust_caption`. | `out.mp4` |
| `unresolvable_tie` | Conflict in a mixed segment → symmetric weights → tie → tiebreaker picks `trust_audio`. | `out_needs_review.mp4` |
| `source_offline_cache_fresh` | Caption service offline, fresh cache → degraded fallback with cached captions. | `out_degraded.mp4` |
| `source_offline_cache_stale` | Caption service offline, stale cache (>1hr) → degraded fallback, staleness logged. | `out_degraded.mp4` |
| `source_offline_cache_empty` | Caption service offline, no cache → export without captions. | `out_no_captions.mp4` |
| `source_offline_cache_signature_mismatch` | Caption service offline, cache signature mismatch → refuse cache, export without captions. | `out_no_captions.mp4` |

---

## Architecture

```
START -> planner (LLM) -> step_executor (code, loops) -> ...
  step_executor routes to:
    - decider (LLM + deterministic cost model) -- on conflict
    - replanner (LLM) -- on failure
    - exporter (ffmpeg) -- when plan complete
  -> audit_writer -> END
```

- **Planner**: LLM authors the step sequence (with hardcoded fallback)
- **Step executor**: deterministic dispatch to tools (fetch, detect, cost, export)
- **Decider**: LLM reads pre-computed CostReport and writes a verdict; code
  overlays tie detection and validates the choice against the cost model
- **Replanner**: LLM produces a new plan on failure; code validates that the
  new plan follows the strategy rules for the specific failure mode
- **Exporter**: ffmpeg renders the resolved timeline
- **Audit writer**: every transition logged to JSONL

See `DECISION_RULE.md` for the full decision rule and justification.
See `docs/architecture.md` for the graph diagram and node reference.

---

## Key Design Decisions

1. **Every branching decision is code-computed**, not LLM-opinion. The LLM
   writes rationale; code computes costs, ties, tiebreaker choices, and
   cache strategy.

2. **Planner failure is terminal** — if the LLM can't produce a valid plan, the agent stops and refuses to export. No silent substitution of a hardcoded plan. If the LLM is unavailable
   (LM Studio offline, Groq key missing, parse failure), the agent completes
   the run using `DEFAULT_PLAN`, `DETERMINISTIC_DECIDER`, and
   `DEFAULT_REPLAN_*` constants. The audit log records which fallback fired.

3. **Cache integrity via SHA256 signature**. When the caption service is
   offline, the agent computes a partial-file hash of the video and compares
   it to the cache's stored signature. Mismatch → refuse cache → export
   without captions. Shipping no subtitles is honest; shipping misaligned
   ones is not.

4. **Canonical plans enforced in code**. The replanner LLM can choose a
   strategy, but the tool list for each strategy is canonical (hardcoded).
   The LLM keeps its rationale; code owns the execution path.

---

## What I Would Do With More Time

- **Exporter failure path**: currently if ffmpeg exits non-zero, the run ends
  with stderr in the audit log. With more time I'd add a `replanner`-style
  edge from `exporter` that tries a fallback codec/preset.

- **Both sources offline**: if both caption and audio services are down, the
  agent should refuse to export and write a clean error. Currently defined as
  `strategy="abort"` but no scenario triggers it.

- **Real speech+music test clip**: the synthetic asset is a blue background
  with a synthesized music bed. With more time I'd add an `--input-video`
  flag support for licensed stock footage (Pexels/Pixabay) to demonstrate
  the trade-offs on real audio.

- **Web UI for the audit log**: a simple Flask/Streamlit page that reads
  `logs/run_<ts>.jsonl` and renders the decision trace as a timeline.

- **Batched multi-conflict LLM call**: currently the decider handles multiple
  conflicts in a single structured-output call. With more time I'd add
  per-conflict reasoning depth and conflict-dependency analysis (does
  resolving c2/d2 affect c3/d3?).

- **Cross-source sanity check**: if cached captions end at 47s but live ducks
  end at 11s, the captions clearly don't belong to this video even without a
  signature. A second-layer integrity check based on content overlap.

---

## Repository Structure

```
agent/                 LangGraph agent (state, cost model, detector, nodes, graph, ...)
services/              FastAPI timing sources (caption + audio mixer)
scripts/               Asset generation, demo runner, scenario definitions
assets/                Generated videos (sample.mp4, conflict_demo.mp4, out*.mp4)
logs/                   JSONL audit traces (run_<timestamp>.jsonl)
tests/                  Pytest suite (39 tests)
DECISION_RULE.md        Explicit decision rule + justification
README.md               This file
```

---

## License

Built for the LEC AI Engineering Intern assessment. All code is original.
The synthetic video asset contains no copyrighted material.