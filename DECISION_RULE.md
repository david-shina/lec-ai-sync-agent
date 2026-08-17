# Decision Rule

This document explains the decision rule the agent uses to reconcile conflicting
video timings between the caption service and the audio mixer, and **why** each
choice was made.

---

## The Problem

Two independent systems report when events should happen in a video:

- **Caption service**: when subtitles should appear and disappear
- **Audio mixer**: when music should duck (volume reduction) to make room for speech

These systems sometimes disagree. For example, the caption service says a subtitle
should start at 5.2 seconds, but the audio mixer has already committed to ducking
at 5.0 seconds. The agent must decide which constraint wins, and justify that
choice.

---

## The Cost Formula

The agent does not "just pick one." It computes a numeric cost for each of three
resolution options, weights those costs by video metadata (segment type), and picks
the lowest-cost option.

### Three options

| Option | What it does | What it costs |
|--------|-------------|---------------|
| A. `trust_caption` | Move the duck to match the caption | Audio drift (the duck shifts away from where the mixer committed) |
| B. `trust_audio` | Move the caption to match the duck | Caption drift (the subtitle shifts away from where the service reported) |
| C. `re_render_duck` | Re-mix the audio so the duck falls exactly at the caption time | A large re-render penalty (compute + audible artifact risk) |

### Drift formula

```
drift = |start_diff| + 0.5 * |end_diff|
```

Start divergence is weighted 2x more than end divergence because a caption that
appears late is more perceptually jarring than one that lingers a bit long.

### Cost formula

```
cost(option) = w_caption * caption_drift(option)
             + w_audio   * audio_drift(option)
             + w_rerender * R * rerender(option)
```

where:

- `caption_drift(trust_caption) = 0` (captions stay put)
- `audio_drift(trust_caption) = drift` (ducks shift to match)
- `audio_drift(trust_audio) = 0` (ducks stay put)
- `caption_drift(trust_audio) = drift` (captions shift to match)
- `rerender(re_render_duck) = 1`, `rerender(others) = 0`
- `R = 8.0` (the re-render penalty constant)

### Weight table (driven by segment metadata)

| Segment type | w_caption | w_audio | w_rerender | Rationale |
|-------------|-----------|---------|-----------|----------|
| **dialogue** | 1.0 | 0.4 | 1.0 | In speech, subtitles are the primary access path. A misaligned subtitle is jarring; a slightly-off music dip is less noticeable. |
| **music** | 0.3 | 1.0 | 1.0 | In music-heavy segments, audio continuity is the foreground. Audio drift costs 3.3x more than caption drift. |
| **mixed** | 0.7 | 0.7 | 1.0 | Both signal and music carry meaning. Symmetric weights are **deliberate**: they make ties possible, triggering Failure B. |

### Why R = 8.0

Re-rendering audio means invoking the audio mixer to compute a new duck envelope.
This costs compute, risks audible artifacts at the duck edges, and is irreversible
without another mix pass. `R = 8.0` means "re-rendering is as costly as 8 seconds
of perceptual drift" — roughly 40x the largest realistic single-conflict drift
(~0.2-0.5 s). The re-render option never wins on cost; it's reserved for explicit
policy decisions.

---

## Tie Detection (deterministic, not LLM opinion)

A tie is declared **in code**, not by the LLM:

```
computed_tie = (marginal_advantage < EPS_TIE) OR (segment_type == "mixed")
```

- `marginal_advantage` = sorted(costs)[1] - sorted(costs)[0] (gap between 2nd-best and best)
- `EPS_TIE = 0.05` (50 ms in cost-units = seconds-equivalent)

Two routes into a tie:
1. **Numeric route**: the drift is so small that both options have nearly equal cost
   (e.g., 0.05 s drift in a dialogue segment → costs 0.02 vs 0.03 → marginal 0.01 < 0.05)
2. **Metadata route**: the conflict sits in a "mixed" segment where symmetric weights
   (0.7/0.7) make the costs structurally identical regardless of drift magnitude

The LLM's verdict is **overridden** by `computed_tie` in code. The LLM's rationale
is preserved in the audit log, but the branching decision is always numeric.

---

## Tiebreaker Policy (when costs are tied)

When `computed_tie == True`, the agent does NOT silently pick. It escalates via the
**replanner** node, which applies a deterministic tiebreaker:

```
apply_tiebreaker(cost_report):
    among tied options, prefer rerender == 0
    if both have rerender == 0, prefer "trust_audio" (preserve audio continuity)
    "re_render_duck" NEVER wins the tiebreaker
    output flagged as needs_review = True
```

**Policy name**: `min_rerender_then_audio_continuity`

**Why this policy**:
- Re-rendering is expensive and risks audible artifacts → prefer zero-rerender options
- Between two zero-rerender options, audio continuity is cheaper to preserve (no
  re-mix needed) → prefer `trust_audio`
- The output is flagged `needs_review` because the choice was policy-driven, not
  data-driven — a human should verify the decision

The policy, both tied options, their costs, and the tiebreaker rationale are all
recorded in the JSONL audit log.

---

## Failure Mode A: Source Offline

### Trigger
`fetch_caption_source()` or `fetch_audio_source()` raises `SourceOffline` after a
3-second HTTP timeout.

### Cache integrity check (three-tier)

Before the replanner decides strategy, `query_cache_state()` runs:

| Cache state | Strategy | Output | Flag |
|------------|----------|--------|------|
| `valid` + fresh | `fallback_cache` | `out_degraded.mp4` | `degraded_mode=true` |
| `valid` + stale (>1 hr) | `fallback_stale_cache` | `out_degraded.mp4` | `degraded_mode=true`, staleness logged |
| `absent` | `export_no_captions` | `out_no_captions.mp4` | `degraded_mode=true`, `needs_review=true` |
| `wrong_video_id` | `export_no_captions` | `out_no_captions.mp4` | (same) |
| `signature_mismatch` | `export_no_captions` | `out_no_captions.mp4` | (same) |

### Cache signature

```python
SHA256(file_size + first_64KB + last_64KB)
```

Partial-file hash. Detects: different video, re-encoded video, or cache from a
different project. Fast on any file size. If the signature does not match, the
cache is rejected — **shipping no subtitles is honest; shipping misaligned ones is not**.

### Why three tiers instead of "just use the cache"

- **Fresh cache**: safe to use, mark degraded
- **Stale cache**: still usable but the staleness is flagged in the audit log
- **No cache / corrupt cache**: refusing to fabricate captions that might belong to
  a different video is the only honest move when the source is down and the cache
  cannot be verified

---

## Failure Mode B: Unresolvable Conflict

### Trigger
`computed_tie == True` (see Tie Detection above).

### Agent behavior
1. `decider` yields the verdict (`chosen_option = null`, `tie = true`)
2. Conditional edge routes to `replanner` with `failure_mode = "unresolvable_tie"`
3. `replanner` produces a plan that must contain exactly: `apply_tiebreaker` + `export_video`
4. `apply_tiebreaker` (deterministic code) picks the winner per the policy above
5. `export_video` renders to `out_needs_review.mp4`
6. `needs_review = true` and `chosen: trust_audio` recorded in audit log

The replanner **never retries the decider** — retrying the same step is explicitly
warned against in the brief. It changes strategy entirely: from "decide on the
merits" to "apply a documented policy."

---

## What the LLM Does vs. What Code Does

| Component | LLM? | Code overlay |
|----------|------|-------------|
| Planner step order | Yes (structured output, Groq primary) | Falls back to `DEFAULT_PLAN` if LLM fails |
| Cost computation | No | `evaluate_option_costs()` — pure deterministic |
| Tie detection | No | `detect_unresolvable()` — pure deterministic |
| Verdict choosing option | Yes (structured output) | Validated against cost report; corrected if it disagrees |
| Verdict `tie` flag | **No** (LLM's value is advisory) | `computed_tie` overrides in code |
| Verdict rationale text | Yes | Preserved in audit log unconditionally |
| Tiebreaker choice | **No** | `apply_tiebreaker()` — pure deterministic |
| Cache strategy | Yes (structured output) | Validated against `cache_state`; canonical plan enforced |
| Replan tool list | Yes (structured output) | Validated against `ALLOWED_TOOLS_BY_FAILURE`; corrected if forbidden tools included |
| Export | No | `ffmpeg` subprocess |

**Headline property**: every branching decision and every final choice is computed
by code. The LLM writes defensible human-readable rationale strings. If the LLM
disagrees with the cost model, its verdict is silently corrected and `_corrected=true`
is logged in the audit trail.

---

## Audit Trail

Every state transition is written to `logs/run_<timestamp>.jsonl` — one JSON object
per transition. A reviewer can read this file and reconstruct the entire run:
which steps ran, what the LLM said, what code computed, whether any correction was
applied, and what the final output was. Each node-to-node routing decision is also
recorded as a `{"event": "route", "from", "to", "reason"}` row, so the branching
between nodes is visible without reading the graph code. Run
`python scripts/run_demo.py --scenario conflict --print-log` for a human-readable
trace of the same run.

A committed sample trace lives at `logs/example_run.jsonl` so reviewers can inspect
a real run without running the repo.
---

## Planner Failure: Refuse to Ship

If the planner LLM cannot produce a valid plan (LLM call fails, JSON schema
validation fails, or the plan is rejected by `validate_plan()`), the agent
**stops entirely**. No fallback plan is substituted. No video is exported.

The graph routes from `planner` directly to `audit_writer` → `END`,
recording the exact failure reason in the audit log. The run exits with
a non-zero exit code and prints "RUN ABORTED: planner could not produce
a valid plan."

This is a deliberate design choice: shipping a video using a hardcoded
plan the LLM never produced would be dishonest. The agent refuses to
export rather than silently substitute its own judgment for the LLM's.