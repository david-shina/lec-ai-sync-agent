from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, model_validator

from config import RERENDER_PENALTY, EPS_TIE



#################################### RAW TIMING DATA MODELS ####################################
class Cue(BaseModel):
    """
    Represents a single caption cue in the video."""
    id: str
    text: str
    start: float
    end: float


class Duck(BaseModel):
    """
    Represents a single music duck in the video.
    """
    id: str
    start: float
    end: float
    gain_reduction_db: float = -12.0


class DuckWindowDict(TypedDict):
    id: str
    start: float
    end: float
    gain_reduction_db: float

#############################################################################################


################################# VIDEO SEGMENT MODELS ##################################

class Segment(BaseModel):
    """
    Represents a segment of the video, which can be a dialogue, music, or mixed segment"""
    start: float
    end: float
    type: str        


class VideoMetadata(BaseModel):
    """Result of querying the video file for its metadata, using ffprobe"""
    duration: float
    segments: list[Segment]


#############################################################################################


################################# CONFLICT AND COST MODELS ##################################

class Conflict(BaseModel):
    """
    Represents a conflict between a caption and a music duck (timing overlap)"""
    caption_id: str
    duck_id: str
    caption_start: float
    duck_start: float
    caption_end: float
    duck_end: float
    start_diff: float
    end_diff: float
    overlap_seconds: float
    kind: str = "start_misalignment"


class CostOption(BaseModel):
    """
    Represents a single option for resolving a conflict, along with its associated cost and breakdown"""
    option: str               # "trust_caption" | "trust_audio" | "re_render_duck"
    cost: float
    breakdown: dict[str, Any]


class CostReport(BaseModel):
    """
    Represents the cost evaluation for one conflict. This is the deterministic input the decider LLM reads
    """
    conflict_id: str
    segment_type: str
    weights: dict[str, float]
    options: list[CostOption]
    best: str
    marginal_advantage: float
    rerender_penalty_constant_R: float = RERENDER_PENALTY
    tie_rule: str = "computed_tie = (marginal_advantage < {0}) OR (segment_type == 'mixed')".format(EPS_TIE)


class Verdict(BaseModel):
    """
    Represents the decision for one conflict. This is where the LLM contributes reasoning, and where code overlays enforce correctness.
    """
    conflict_id: str
    chosen_option: Optional[str] = None
    rationale: str = ""
    cost_breakdown: dict[str, Any] = Field(default_factory=dict)
    confidence: str = "high"        # "high" | "medium" | "tie"
    tie: bool = False
    source: str = "llm"
    llm_claimed_tie: Optional[bool] = None
    corrected: bool = False

#############################################################################################

################################ AGENT STATE MODELS #########################################

class Step(BaseModel):
    """
    Represents a single step in the agent's plan"""
    tool: str
    intent: str = ""    


class Plan(BaseModel):
    """
    Represents the agent's plan, which is a sequence of steps to achieve the goal"""
    steps: list[Step]
    rationale: str = ""
    source: str = "llm"
    corrected: bool = False

    @model_validator(mode="after")
    def must_end_with_export_video(self) -> "Plan":
        if not self.steps:
            raise ValueError("Plan must contain at least one step")
        if self.steps[-1].tool != "export_video":
            raise ValueError(f"Plan must end with export_video (last step is {self.steps[-1].tool})")
        return self


class Replan(BaseModel):
    """
    Represents a replan, which is a new sequence of steps to achieve the goal after a failure or change in circumstances"""
    new_plan: list[Step]
    strategy: str
    rationale: str = ""
    source: str = "llm"
    corrected: bool = False


class AgentState(TypedDict, total=False):
    goal: str
    video_id: str
    video_path: str
    scenario: str

    plan: list[dict]
    step_idx: int
    observations: list[dict]

    captions: Optional[list[dict]]
    ducks: Optional[list[dict]]
    metadata: Optional[dict]
    conflicts: Optional[list[dict]]
    conflicts_resolved: bool
    segment_overrides: Optional[dict]
    cost_report: Optional[dict]
    cost_reports_by_conflict: Optional[dict[str, dict]]
    tiebreaker_result: Optional[dict]
    cost_reports_by_conflict: Optional[dict[str, dict]]
    cache_state: Optional[dict]

    verdict: Optional[dict]
    verdicts: Optional[list[dict]]
    resolved_timeline: Optional[dict]
    chosen_options_by_conflict: Optional[dict[str, str]]

    failure_mode: Optional[str]
    failed_step: Optional[dict]
    degraded_mode: bool
    planner_failed: bool
    planner_failure_reason: Optional[str]
    needs_review: bool
    output_path: Optional[str]
    output_suffix: str
    export_success: Optional[bool]

    audit: list[dict]
    run_id: str




class Verdicts(BaseModel):
    verdicts: list[Verdict]

#############################################################################################
