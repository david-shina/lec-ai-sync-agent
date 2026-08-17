from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from config import (
    LM_STUDIO_BASE_URL, LM_STUDIO_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    DECIDER_MAX_TOKENS,
)

log = logging.getLogger(__name__)


def _groq_kwargs(max_tokens: int = 1000) -> dict:
    return {
        "temperature": 0,
        "max_tokens": max_tokens,
        "request_timeout": 30.0,
        "timeout": 30.0,
    }


def _lm_studio_kwargs(max_tokens: int = 600) -> dict:
    return {
        "base_url": LM_STUDIO_BASE_URL,
        "api_key": "lm-studio",
        "model": LM_STUDIO_MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "request_timeout": 30.0,
        "timeout": 30.0,
    }


def _use_groq() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "PASTE_YOUR_KEY_HERE"


def get_planner_llm():
    if _use_groq():
        log.info("planner using Groq (%s)", GROQ_MODEL)
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, **_groq_kwargs(400))
    log.info("GROQ_API_KEY not set; planner falling back to LM Studio")
    return ChatOpenAI(**_lm_studio_kwargs(400))


def get_replanner_llm():
    if _use_groq():
        log.info("replanner using Groq (%s)", GROQ_MODEL)
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, **_groq_kwargs(600))
    log.info("GROQ_API_KEY not set; replanner falling back to LM Studio")
    return ChatOpenAI(**_lm_studio_kwargs(600))


def get_decider_llm():
    """Returns the LLM for the decider node. No tools bound - cost reports
    are pre-computed and passed directly in the prompt text."""
    if _use_groq():
        log.info("decider using Groq (%s)", GROQ_MODEL)
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, **_groq_kwargs(DECIDER_MAX_TOKENS))
    log.info("GROQ_API_KEY not set; decider falling back to LM Studio")
    return ChatOpenAI(**_lm_studio_kwargs(DECIDER_MAX_TOKENS))


def is_offline_friendly() -> bool:
    return not _use_groq()

def current_llm_info() -> dict:
    """Returns dict with {backend, model} of whichever LLM is currently active."""
    if _use_groq():
        return {"backend": "groq", "model": GROQ_MODEL}
    return {"backend": "lm_studio", "model": LM_STUDIO_MODEL}
