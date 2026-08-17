from __future__ import annotations

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command

from agent.state import (
    AgentState, Plan, Replan, Step, Verdict,
    Conflict, VideoMetadata, Verdicts,
)
from agent.planner_defaults import (
    DEFAULT_PLAN, DEFAULT_REPLAN_TIE, DEFAULT_REPLAN_NO_CAPTIONS,
)
from agent.validation import (
    validate_plan, validate_replan_against_cache_state,
    validate_replan_against_failure_mode, validate_verdict_against_cost_report,
    PlanValidationError, ReplanValidationError,
)
from agent.decider_fallback import deterministic_verdicts
from agent.tools import dispatch_tool, SourceOffline
from agent.cost_model import evaluate_option_costs as _evaluate_option_costs
from agent.llm import get_planner_llm, get_replanner_llm, get_decider_llm, current_llm_info
from agent.prompts import (
    PLANNER_SYSTEM, planner_user,
    DECIDER_SYSTEM, decider_user,
    REPLANNER_OFFLINE_SYSTEM, REPLANNER_TIE_SYSTEM, replanner_user,
)
from agent.audit import write_audit_entry
from config import ASSETS_DIR


log = logging.getLogger(__name__)
progress_log = logging.getLogger("demo.progress")


def _conflict_id(c: Conflict | dict) -> str:
    if isinstance(c, dict):
        return f"{c.get('caption_id')}/{c.get('duck_id')}"
    return f"{c.caption_id}/{c.duck_id}"


def _audit(state: dict, node: str, row: dict) -> None:
    transition_idx = len(state.get("audit", []))
    write_audit_entry(state, node, row, transition_idx)


def _log_route(state: dict, from_node: str, to_node: str, reason: str) -> None:
    """Record a node-to-node routing decision in the audit log and progress output."""
    _audit(state, from_node, {
        "event": "route",
        "from": from_node,
        "to": to_node,
        "reason": reason,
    })
    progress_log.info("[ROUTE] %s -> %s (%s)", from_node, to_node, reason)


def _log_replan(state: dict, replan: Replan) -> None:
    progress_log.info("[REPLANNER] REPLAN (strategy=%s, source=%s):", replan.strategy, replan.source)
    for i, s in enumerate(replan.new_plan, start=1):
        progress_log.info("[REPLANNER]   %d. %s - %s", i, s.tool, s.intent or "")
    if replan.rationale:
        progress_log.info("[REPLANNER] Rationale: %s", replan.rationale[:120])


def _summary(state: dict) -> dict:
    return {
        "step_idx": state.get("step_idx"),
        "plan_len": len(state.get("plan", [])),
        "verdicts": bool(state.get("verdicts")),
        "degraded_mode": state.get("degraded_mode", False),
        "needs_review": state.get("needs_review", False),
    }


def planner_node(state: AgentState) -> dict:
    _audit(state, "planner", {"no_op_before": True})
    progress_log.info("[PLANNER] Planning...")
    try:
        llm = get_planner_llm()
        # Use json_mode (more lenient than json_schema) for planner
        plan_obj = llm.with_structured_output(Plan, method="json_mode", include_raw=True).invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=planner_user(state)),
        ])
        plan: Plan = plan_obj["parsed"]
        
        # If parsed is None, try to parse raw response manually
        if plan is None:
            raw = plan_obj.get("raw")
            if raw and hasattr(raw, 'content'):
                import json as _json
                try:
                    parsed_dict = _json.loads(raw.content)
                    plan = Plan(**parsed_dict)
                except Exception:
                    pass
        
        if plan is None:
            raise ValueError("LLM failed to produce valid plan JSON")
        
        validate_plan(plan)
        info = current_llm_info()
        progress_log.info("[PLANNER] Planned %d steps (source=%s, model=%s)", len(plan.steps), info["backend"], info["model"])
        for i, s in enumerate(plan.steps, start=1):
            progress_log.info("[PLANNER]   %d. %s - %s", i, s.tool, s.intent or "")
        if plan.rationale:
            progress_log.info("[PLANNER] Rationale: %s", plan.rationale[:120])
        _audit(state, "planner", {"source":"llm","plan":[s.model_dump() for s in plan.steps],"rationale":plan.rationale})
        return {"plan": [s.model_dump() for s in plan.steps], "step_idx": 0,
                "planner_failed": False, "planner_failure_reason": None}
    except Exception as e:
        log.error("planner FAILED: %s", e)
        progress_log.info("[PLANNER] FAILED: %s", str(e)[:200])
        _audit(state, "planner", {"source":"failed","reject_reason":str(e),"plan":None})
        return {"plan": [], "step_idx": 0,
                "planner_failed": True, "planner_failure_reason": str(e)}
    

def step_executor_node(state: AgentState) -> dict:
    plan = state.get("plan", [])
    idx = state.get("step_idx", 0)
    if idx >= len(plan):
        _audit(state, "step_executor", {"step":None,"result":"no_more_steps"})
        return {}

    step = plan[idx]
    tool_name = step.get("tool") if isinstance(step, dict) else step.tool
    plan_len = len(plan)

    progress_log.info("[EXEC] Step %d/%d: %s", idx + 1, plan_len, tool_name)
    try:
        result = dispatch_tool(tool_name, state)
        brief = "done"
        if "captions" in result:
            brief = str(len(result["captions"])) + " captions fetched"
        elif "ducks" in result:
            brief = str(len(result["ducks"])) + " ducks fetched"
        elif "duration" in result:
            seg_count = len(result.get("segments", []))
            brief = str(round(result["duration"],1)) + "s, " + str(seg_count) + " segments"
        elif "conflicts" in result:
            brief = str(len(result["conflicts"])) + " conflicts detected"
        elif "output_path" in result:
            ok = result.get("success", False)
            brief = ("OK: " if ok else "FAILED: ") + str(result.get("output_path",""))
        elif "chosen" in result:
            brief = "chose " + str(result.get("chosen"))
        elif "state" in result:
            brief = "cache_state=" + str(result.get("state"))
        progress_log.info("[EXEC] Step %d/%d: %s -> %s", idx + 1, plan_len, tool_name, brief)
        _audit(state, "step_executor", {
            "step": idx, "tool": tool_name, "result": result, "plan_len": len(plan),
        })
        update = {"step_idx": idx + 1}

        if tool_name == "fetch_caption_source" and "captions" in result:
            update["captions"] = result["captions"]
        elif tool_name == "fetch_audio_source" and "ducks" in result:
            update["ducks"] = result["ducks"]
        elif tool_name == "query_video_metadata" and "duration" in result:
            update["metadata"] = result
        elif tool_name == "query_metadata_cache" and "captions" in result:
            update["captions"] = result["captions"]
        elif tool_name == "query_cache_state":
            update["cache_state"] = result
        elif tool_name == "detect_conflicts" and "conflicts" in result:
            update["conflicts"] = result["conflicts"]
        elif tool_name == "apply_tiebreaker":
            # Handle both single-conflict and multi-conflict tiebreaker results
            if "chosen_by_conflict" in result:
                # Multi-conflict result
                existing = dict(state.get("chosen_options_by_conflict") or {})
                for cid, chosen in result.get("chosen_by_conflict", {}).items():
                    existing[cid] = chosen
                update["chosen_options_by_conflict"] = existing
            else:
                # Single conflict result
                chosen = result.get("chosen")
                cid = result.get("conflict_id")
                if chosen and cid:
                    existing = dict(state.get("chosen_options_by_conflict") or {})
                    existing[cid] = chosen
                    update["chosen_options_by_conflict"] = existing
            update["tiebreaker_result"] = result
        elif tool_name == "export_video":
            update["output_path"] = result.get("output_path")
            update["export_success"] = result.get("success", False)

        obs = state.setdefault("observations", [])
        obs.append({"step": idx, "tool": tool_name, "result": result})
        return update

    except SourceOffline as e:
        _audit(state, "step_executor", {
            "step": idx, "tool": tool_name,
            "error": "source_offline", "source": e.source, "detail": e.detail,
        })
        obs = state.setdefault("observations", [])
        obs.append({"step": idx, "tool": tool_name, "error": "source_offline",
                    "source": e.source, "detail": e.detail})
        log.info("[EXEC] %d/%d %s -> OFFLINE: %s", idx + 1, len(state.get("plan", [])), tool_name, e.source)

        try:
            cs = dispatch_tool("query_cache_state", state)
            _audit(state, "step_executor", {
                "auto_action": "query_cache_state",
                "result": cs,
            })
        except Exception as e2:
            cs = {"state": "absent", "reason": str(e2)}
            _audit(state, "step_executor", {
                "auto_action": "query_cache_state_failed",
                "result": cs,
            })

        which = "caption_service" if e.source == "caption_service" else "audio_mixer"
        progress_log.info("[EXEC] Step %d/%d: %s -> OFFLINE (%s)", idx + 1, plan_len, tool_name, which)
        return {
            "failure_mode": "source_offline",
            "failed_step": {"step": idx, "tool": tool_name, "source": which},
            "cache_state": cs if e.source == "caption_service" else state.get("cache_state"),
            "step_idx": idx,
        }

    except Exception as e:
        _audit(state, "step_executor", {
            "step": idx, "tool": tool_name,
            "error": "tool_exception", "detail": str(e),
        })
        obs = state.setdefault("observations", [])
        obs.append({"step": idx, "tool": tool_name, "error": "tool_exception", "detail": str(e)})
        return {"step_idx": idx + 1}


def decider_node(state: AgentState) -> dict:
    metadata = VideoMetadata(**state["metadata"])
    conflicts = [Conflict(**c) for c in state["conflicts"]]

    cost_reports_by_conflict = {}
    for c in conflicts:
        try:
            cr = _evaluate_option_costs(c, metadata)
            cost_reports_by_conflict[cr.conflict_id] = cr
        except Exception as e:
            log.error("cost model failed for %s: %s", _conflict_id(c), e)
    if not cost_reports_by_conflict:
        _audit(state, "decider", {"error":"cost_model_empty","source":"deterministic_fallback"})
        verdicts = deterministic_verdicts(conflicts, {}, metadata) if False else []
        return {"verdicts": [], "verdict": None}

    cost_reports_dicts = {k: v.model_dump() for k, v in cost_reports_by_conflict.items()}
    state["cost_reports_by_conflict"] = cost_reports_dicts
    state["cost_report"] = list(cost_reports_dicts.values())[0]
    _audit(state, "decider", {
        "pre_computed_cost_reports": cost_reports_dicts,
    })

    progress_log.info("[DECIDER] Resolving %d conflict(s)...", len(conflicts))
    try:
        llm = get_decider_llm()
        structured = llm.with_structured_output(Verdicts, method="json_schema")
        ui_state_for_llm = {
            **state,
            "conflicts": state.get("conflicts", []),
            "metadata": state.get("metadata", {}),
        }
        ut = decider_user(ui_state_for_llm) + (
            "\n\nPRE-COMPUTED COST REPORTS (for reference; the deterministic cost "
            "model produced these; you need not call the tool if the breakdown "
            "is already visible):\n" + json.dumps(cost_reports_dicts, indent=2)
        )
        verdicts_obj = structured.invoke([
            SystemMessage(content=DECIDER_SYSTEM),
            HumanMessage(content=ut),
        ])
        verdicts = list(verdicts_obj.verdicts) if verdicts_obj else []
        _audit(state, "decider", {"source":"llm","raw_verdicts":[v.model_dump() for v in verdicts]})
    except Exception as e:
        log.warning("decider LLM failed (%s). Deterministic fallback.", e)
        verdicts = deterministic_verdicts(conflicts, cost_reports_by_conflict, metadata)
        _audit(state, "decider", {"source":"deterministic_fallback","error":str(e)})

    final_verdicts: list[Verdict] = []
    any_tie = False
    chosen_by_conflict: dict[str, str | None] = {}
    for v in verdicts:
        cr = cost_reports_by_conflict.get(v.conflict_id)
        c_match = next((c for c in conflicts if _conflict_id(c) == v.conflict_id), conflicts[0])
        if cr is None:
            _audit(state, "decider", {"error":"missing_cost_report","conflict_id":v.conflict_id})
            v.tie = True
            v.chosen_option = None
            v.corrected = True
            v.source = "corrected_missing_cost_report"
        else:
            v = validate_verdict_against_cost_report(v, cr, metadata, c_match)
        if v.tie:
            any_tie = True
        chosen_by_conflict[v.conflict_id] = v.chosen_option
        final_verdicts.append(v)

    _audit(state, "decider", {
        "validated_verdicts": [v.model_dump() for v in final_verdicts],
        "any_tie": any_tie,
    })

    info = current_llm_info()
    verdict_summary = ", ".join(f"{v.conflict_id}:{v.chosen_option}" for v in final_verdicts)
    for v in final_verdicts:
        cr = cost_reports_by_conflict.get(v.conflict_id)
        if cr is not None:
            costs = " | ".join(f"{o.option}={o.cost}" for o in cr.options)
            progress_log.info("[DECIDER]   %s -> %s  [%s]", v.conflict_id, v.chosen_option, costs)
        else:
            progress_log.info("[DECIDER]   %s -> %s", v.conflict_id, v.chosen_option)
    progress_log.info("[DECIDER] Done: %s (source=%s)", verdict_summary, info["backend"])

    update: dict = {
        "verdicts": [v.model_dump() for v in final_verdicts],
        "chosen_options_by_conflict": chosen_by_conflict,
        "cost_report": list(cost_reports_by_conflict.values())[0].model_dump()
            if cost_reports_by_conflict else (state.get("cost_report") or {}),
        "cost_reports_by_conflict": cost_reports_dicts,
    }
    if final_verdicts:
        update["verdict"] = final_verdicts[0].model_dump()
    if any_tie:
        update["failure_mode"] = "unresolvable_tie"
    else:
        update["conflicts"] = None
        update["conflicts_resolved"] = True
    return update


def replanner_node(state: AgentState) -> dict:
    fm = state.get("failure_mode")
    progress_log.info("[REPLANNER] Triggered: %s", fm)
    _audit(state, "replanner", {"trigger_failure_mode": fm})

    if fm == "source_offline":
        try:
            llm = get_replanner_llm()
            replan_obj = llm.with_structured_output(Replan, method="json_schema", include_raw=True).invoke([
                SystemMessage(content=REPLANNER_OFFLINE_SYSTEM),
                HumanMessage(content=replanner_user(state)),
            ])
            replan: Replan = replan_obj["parsed"]
            replan = validate_replan_against_cache_state(
                replan, state.get("cache_state") or {"state": "absent"})
            replan = validate_replan_against_failure_mode(replan, fm)
            _audit(state, "replanner", {
                "source":"llm","strategy":replan.strategy,"plan":[s.model_dump() for s in replan.new_plan],
                "rationale":replan.rationale,"corrected":replan.corrected,
            })
            _log_replan(state, replan)
            return _apply_replan_side_effects(state, replan)
        except Exception as e:
            log.warning("replanner(offline) fallback: %s", e)
            cs = state.get("cache_state") or {"state": "absent"}
            from agent.validation import replan_for_cache_state
            new_plan_dicts, strategy = replan_for_cache_state(cs)
            new_plan = [Step(**s) for s in new_plan_dicts]
            replan = Replan(new_plan=new_plan, strategy=strategy,
                            rationale=f"deterministic fallback: {e}",
                            source="fallback", corrected=True)
            _audit(state, "replanner", {
                "source":"fallback","strategy":strategy,"plan":new_plan_dicts,
                "rationale":"deterministic fallback","corrected":True,"error":str(e),
            })
            _log_replan(state, replan)
            return _apply_replan_side_effects(state, replan)

    if fm == "unresolvable_tie":
        try:
            llm = get_replanner_llm()
            replan_obj = llm.with_structured_output(Replan, method="json_schema", include_raw=True).invoke([
                SystemMessage(content=REPLANNER_TIE_SYSTEM),
                HumanMessage(content=replanner_user(state)),
            ])
            replan: Replan = replan_obj["parsed"]
            replan = validate_replan_against_failure_mode(replan, fm)
            _audit(state, "replanner", {
                "source":"llm","strategy":replan.strategy,
                "plan":[s.model_dump() for s in replan.new_plan],
                "rationale":replan.rationale,"corrected":replan.corrected,
            })
            _log_replan(state, replan)
            return _apply_replan_side_effects(state, replan)
        except Exception as e:
            log.warning("replanner(tie) fallback: %s", e)
            new_plan = [Step(**s) for s in DEFAULT_REPLAN_TIE]
            replan = Replan(new_plan=new_plan, strategy="escalate_needs_review",
                            rationale=f"deterministic fallback: {e}",
                            source="fallback", corrected=True)
            _audit(state, "replanner", {
                "source":"fallback","strategy":"escalate_needs_review",
                "plan":DEFAULT_REPLAN_TIE,"corrected":True,"error":str(e),
            })
            _log_replan(state, replan)
            return _apply_replan_side_effects(state, replan)

    _audit(state, "replanner", {"error":"no_failure_mode","strategy":"abort"})
    return {}


def _apply_replan_side_effects(state: dict, replan: Replan) -> dict:
    update: dict = {
        "plan": [s.model_dump() for s in replan.new_plan],
        "step_idx": 0,
    }
    if replan.strategy == "fallback_cache":
        update["degraded_mode"] = True
        update["output_suffix"] = "_degraded"
    elif replan.strategy == "fallback_stale_cache":
        update["degraded_mode"] = True
        update["output_suffix"] = "_degraded"
    elif replan.strategy == "export_no_captions":
        update["degraded_mode"] = True
        update["needs_review"] = True
        update["output_suffix"] = "_no_captions"
        update["no_captions"] = True
        update["conflicts"] = None
        update["verdicts"] = None
        update["chosen_options_by_conflict"] = {}
    elif replan.strategy == "escalate_needs_review":
        update["needs_review"] = True
        update["output_suffix"] = "_needs_review"
        update["conflicts"] = None
        update["conflicts_resolved"] = True
    elif replan.strategy == "abort":
        update["output_suffix"] = "_aborted"
        update["needs_review"] = True
    _log_route(state, "replanner", "step_executor", f"replan complete, strategy={replan.strategy}")
    return update


def audit_writer_node(state: AgentState) -> dict:
    if state.get("planner_failed"):
        _audit(state, "audit_writer", {
            "summary": {"planner_failed": True,
                        "reason": state.get("planner_failure_reason")},
            "verdicts": None,
            "output_path": None,
            "export_success": False,
            "chosen_options_by_conflict": {},
        })
        progress_log.info("[EXPORT] Aborted - planner failed")
        return {}

    _audit(state, "audit_writer", {
        "summary": _summary(state),
        "verdicts": state.get("verdicts") or None,
        "output_path": state.get("output_path"),
        "export_success": state.get("export_success"),
        "chosen_options_by_conflict": state.get("chosen_options_by_conflict") or {},
    })
    if state.get("export_success"):
        progress_log.info("[EXPORT] %s", state.get("output_path"))
    else:
        progress_log.info("[EXPORT] No output (export_success=False)")
    return {}


def route_after_planner(state: AgentState) -> str:
    if state.get("planner_failed"):
        _log_route(state, "planner", "audit_writer",
                   f"planner failed: {str(state.get('planner_failure_reason'))[:80]}")
        return "audit_writer"
    _log_route(state, "planner", "step_executor",
               f"plan ready ({len(state.get('plan', []))} steps)")
    return "step_executor"

def route_after_step_executor(state: AgentState) -> str:
    obs = state.get("observations") or []
    last = obs[-1] if obs else None
    if last and last.get("error") == "source_offline":
        _log_route(state, "step_executor", "replanner",
                   f"source offline ({last.get('source')})")
        return "replanner"
    if (state.get("failure_mode") == "source_offline"
            and state.get("step_idx", 0) == 0
            and not state.get("_replanner_done")):
        state["_replanner_done"] = True
        _log_route(state, "step_executor", "replanner", "source offline (replan pending)")
        return "replanner"
    if state.get("conflicts") and not state.get("conflicts_resolved"):
        _log_route(state, "step_executor", "decider",
                   f"{len(state['conflicts'])} conflicts unresolved")
        return "decider"
    if state.get("step_idx", 0) >= len(state.get("plan", [])):
        _log_route(state, "step_executor", "audit_writer", "all steps complete")
        return "audit_writer"
    _log_route(state, "step_executor", "step_executor",
               f"continue plan (step {state.get('step_idx', 0) + 1}/{len(state.get('plan', []))})")
    return "step_executor"


def route_after_decider(state: AgentState) -> str:
    verdicts = state.get("verdicts") or []
    any_tie = any(v.get("tie") for v in verdicts)
    if any_tie:
        _log_route(state, "decider", "replanner", "unresolvable tie")
        return "replanner"
    _log_route(state, "decider", "step_executor", "all conflicts resolved")
    return "step_executor"
