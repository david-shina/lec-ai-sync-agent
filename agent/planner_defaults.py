from __future__ import annotations

from agent.state import Step


ALL_TOOL_VOCAB = [
    "fetch_caption_source",
    "fetch_audio_source",
    "query_video_metadata",
    "query_metadata_cache",
    "query_cache_state",
    "detect_conflicts",
    "evaluate_option_costs",
    "apply_tiebreaker",
    "export_video",
]

DEFAULT_PLAN = [
    {"tool": "fetch_caption_source",   "intent": "fetch caption timings"},
    {"tool": "fetch_audio_source",     "intent": "fetch audio ducking timings"},
    {"tool": "query_video_metadata",   "intent": "classify segments by content type"},
    {"tool": "detect_conflicts",       "intent": "compare caption and duck timings"},
    {"tool": "export_video",           "intent": "render the final video with resolved timeline"},
]

DEFAULT_REPLAN_CACHE_FRESH = [
    {"tool": "query_metadata_cache", "intent": "use cached captions, fresh"},
    {"tool": "fetch_audio_source",   "intent": "audio service is presumed alive"},
    {"tool": "query_video_metadata", "intent": "classify segments"},
    {"tool": "detect_conflicts",     "intent": "compare cached captions vs live ducks"},
    {"tool": "export_video",         "intent": "render degraded output"},
]

DEFAULT_REPLAN_CACHE_STALE = DEFAULT_REPLAN_CACHE_FRESH

DEFAULT_REPLAN_NO_CAPTIONS = [
    {"tool": "fetch_audio_source",   "intent": "audio still alive, fetch duck windows"},
    {"tool": "query_video_metadata", "intent": "classify segments (still useful)"},
    {"tool": "export_video",         "intent": "render audio-only, no subtitles"},
]

DEFAULT_REPLAN_TIE = [
    {"tool": "apply_tiebreaker", "intent": "code picks the tied option per documented policy"},
    {"tool": "export_video",     "intent": "render output flagged for human review"},
]

DEFAULT_REPLAN_ABORT = [
    {"tool": "export_video", "intent": "DO NOT render — write clean error to audit"},
]

ALLOWED_TOOLS_BY_FAILURE = {
    "source_offline": {
        "fetch_audio_source", "query_video_metadata",
        "query_metadata_cache", "query_cache_state",
        "detect_conflicts", "export_video",
    },
    "unresolvable_tie": {
        "apply_tiebreaker", "export_video",
    },
}

FORBIDDEN_TOOLS_BY_FAILURE = {
    "source_offline": {"fetch_caption_source"},
    "unresolvable_tie": {"decider", "evaluate_option_costs", "fetch_caption_source",
                         "fetch_audio_source", "detect_conflicts"},
}
