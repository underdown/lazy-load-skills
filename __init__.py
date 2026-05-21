"""
lazy-load-skills — Pre-load only relevant skills per user message.

Hooks into Hermes's ``pre_llm_call`` lifecycle to intercept every user
message before the LLM sees it, computes skill relevance against the
message content, and injects the full SKILL.md content of only the
top-N most relevant skills as ephemeral context.

This eliminates the need for the LLM to call ``skill_view()`` for those
skills (saving API calls) and makes the massive ``<available_skills>``
block in the system prompt moot (the agent is told to ignore unlisted
skills).

Key optimizations:
• System prompt: 2-5K tokens saved (agent ignores irrelevant skills)
• API calls:    0-N calls saved (pre-loaded skills skip skill_view)
• Latency:      ~0ms added (keyword matching is sub-millisecond)

Usage:
    hermes plugins install lazy-load-skills
    hermes plugins enable lazy-load-skills

Configuration (env vars in ~/.hermes/.env):
    LAZY_SKILLS_TOP_N=5          # max skills to pre-load (default: 5)
    LAZY_SKILLS_METHOD=keyword   # "keyword" or "embedding" (default: keyword)
    LAZY_SKILLS_MIN_SCORE=0.2    # minimum relevance score 0.0-1.0 (default: 0.15)
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .relevance import compute_relevance, get_available_skills
from .skill_loader import load_skill_content, get_skills_dir

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

TOP_N = _env_int("LAZY_SKILLS_TOP_N", 5)
METHOD = os.getenv("LAZY_SKILLS_METHOD", "keyword").lower()
MIN_SCORE = _env_float("LAZY_SKILLS_MIN_SCORE", 0.15)

# ── Per-session cache ─────────────────────────────────────────────────────
# Maps session_id → (last_message_hash, cached_skill_list)
# Avoids recomputing relevance when the user sends follow-up messages
# in the same conversation thread.

_session_cache: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}


def _hash_message(text: str) -> int:
    """Fast hash of the user message for cache invalidation."""
    # Use the first 200 chars + last 50 chars to balance sensitivity
    sample = text[:200] + text[-50:] if len(text) > 250 else text
    return hash(sample)


def _build_skills_context(
    user_message: str,
) -> str:
    """Build the context block that replaces the system prompt skills list.

    Returns a string to inject into the user message context, or empty
    string if no skills are relevant enough.
    """
    # Get all available skills for relevance scoring
    all_skills = get_available_skills()
    if not all_skills:
        return ""

    # Compute relevance
    ranked = compute_relevance(
        user_message=user_message,
        skills=all_skills,
        method=METHOD,
        top_n=TOP_N,
        min_score=MIN_SCORE,
    )

    if not ranked:
        logger.debug("lazy-load-skills: no skills above relevance threshold")
        return ""

    # Load full content for top skills
    loaded_skills = []
    for skill in ranked:
        name = skill["name"]
        content = load_skill_content(name)
        loaded_skills.append({
            "name": name,
            "description": skill.get("description", ""),
            "score": skill.get("score", 0.0),
            "content": content,
        })

    # Build the context block
    parts = []
    parts.append(
        "The following skills are relevant to this conversation. "
        "Their full content has been pre-loaded below — do NOT call "
        "skill_view() for these skills. Ignore all other skills listed "
        "in the system prompt; only these are applicable."
    )
    parts.append("")

    for s in loaded_skills:
        score_pct = int(s["score"] * 100)
        parts.append(
            f"## Skill: {s['name']} (relevance: {score_pct}%)"
        )
        if s["description"]:
            parts.append(f"_{s['description']}_")
        if s["content"]:
            parts.append("")
            parts.append(s["content"])
        parts.append("")

    context = "\n".join(parts)

    logger.debug(
        "lazy-load-skills: loaded %d skills (%.0f chars) for message hash %d",
        len(loaded_skills),
        len(context),
        _hash_message(user_message),
    )

    return context


# ── Hook handler ───────────────────────────────────────────────────────────

def on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    is_first_turn: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, str]]:
    """pre_llm_call hook: inject focused skills context.

    Called by Hermes before every LLM API call. We only act on the first
    message in a session OR when the conversation shifts significantly
    (message hash differs from cache).

    Returns a dict with ``"context"`` key containing the focused skills
    block, which the agent injects into the user message as ephemeral
    context (not persisted to session DB).
    """
    if not user_message or not session_id:
        return None

    msg_hash = _hash_message(user_message)

    # Check cache — skip if same message context
    if session_id in _session_cache:
        cached_hash, _ = _session_cache[session_id]
        if cached_hash == msg_hash and not is_first_turn:
            logger.debug("lazy-load-skills: cache hit for session %s", session_id)
            return None

    # Compute and cache
    context = _build_skills_context(
        user_message=user_message,
    )

    if not context:
        return None

    _session_cache[session_id] = (msg_hash, [])

    return {"context": context}


# ── Plugin entry point ─────────────────────────────────────────────────────

def register(ctx) -> None:
    """Register the pre_llm_call hook with Hermes."""
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info(
        "lazy-load-skills registered (top_n=%d, method=%s, min_score=%.2f)",
        TOP_N, METHOD, MIN_SCORE,
    )
