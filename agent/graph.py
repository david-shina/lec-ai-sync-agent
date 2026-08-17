from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    planner_node, step_executor_node, decider_node, replanner_node,
    audit_writer_node, route_after_planner,
    route_after_step_executor, route_after_decider,
)


def build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("planner", planner_node)
    wf.add_node("step_executor", step_executor_node)
    wf.add_node("decider", decider_node)
    wf.add_node("replanner", replanner_node)
    wf.add_node("audit_writer", audit_writer_node)

    wf.add_edge(START, "planner")
    wf.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "step_executor": "step_executor",
            "audit_writer": "audit_writer",
        },
    )
    wf.add_conditional_edges(
        "step_executor",
        route_after_step_executor,
        {
            "decider": "decider",
            "replanner": "replanner",
            "step_executor": "step_executor",
            "audit_writer": "audit_writer",
        },
    )
    wf.add_conditional_edges(
        "decider",
        route_after_decider,
        {
            "step_executor": "step_executor",
            "replanner": "replanner",
        },
    )
    wf.add_edge("replanner", "step_executor")
    wf.add_edge("audit_writer", END)

    return wf.compile()
