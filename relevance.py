"""
Skill relevance computation.

Supports two methods:
- ``keyword`` (default): Fast TF-IDF-like keyword overlap. No dependencies.
  Sub-millisecond per query. Good enough for 90%+ of cases.
- ``embedding``: Uses sentence-transformers to compute cosine similarity
  between the user message and skill descriptions. More accurate for
  ambiguous queries. Requires ``pip install sentence-transformers``.

Scoring:
  Each skill gets a score 0.0–1.0. Skills below ``min_score`` are filtered.
  The top ``top_n`` skills are returned, ordered by relevance.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Stop words for keyword method ──────────────────────────────────────────

_STOP_WORDS: set = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "you", "your",
    "yours", "i", "me", "my", "mine", "we", "us", "our", "ours", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "just", "very", "too", "also", "only", "then", "now",
    "here", "there", "all", "some", "any", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "about", "up",
    "out", "if", "so", "than", "into", "over", "after", "before",
    "between", "under", "again", "further", "once",
}

# ── Tokenization ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, filter stop words and short tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


# ── Keyword method ─────────────────────────────────────────────────────────

def _keyword_score(query_tokens: List[str], skill_tokens: List[str]) -> float:
    """Simple overlap-based relevance score.

    Returns a score 0.0–1.0 based on:
    - Direct token overlap (weight: 0.6)
    - Partial token overlap — query tokens that are substrings of skill tokens
      or vice versa (weight: 0.3)
    - Token count normalization (weight: 0.1) — penalizes skills with very
      few tokens to prevent generic matches
    """
    if not query_tokens or not skill_tokens:
        return 0.0

    query_set = set(query_tokens)
    skill_set = set(skill_tokens)

    # Direct overlap
    intersection = query_set & skill_set
    direct_score = len(intersection) / max(len(query_set), 1)

    # Partial overlap: query words contained in skill words
    partial_hits = 0
    for qt in query_set:
        if qt not in intersection:
            for st in skill_set:
                if qt in st or st in qt:
                    partial_hits += 1
                    break
    partial_score = partial_hits / max(len(query_set), 1)

    # Normalization: prefer skills with reasonable token counts
    norm_score = min(len(skill_tokens) / 20.0, 1.0)

    return (direct_score * 0.6) + (partial_score * 0.3) + (norm_score * 0.1)


def _keyword_relevance(
    user_message: str,
    skills: List[Dict[str, Any]],
    top_n: int = 5,
    min_score: float = 0.15,
) -> List[Dict[str, Any]]:
    """Rank skills by keyword overlap with the user message."""
    query_tokens = _tokenize(user_message)

    scored = []
    for skill in skills:
        name = skill.get("name", "")
        desc = skill.get("description", "")

        # Build tokens from name + description
        skill_text = f"{name} {desc}"
        skill_tokens = _tokenize(skill_text)

        score = _keyword_score(query_tokens, skill_tokens)

        if score >= min_score:
            scored.append({
                "name": name,
                "description": desc,
                "score": round(score, 4),
            })

    # Sort by score descending
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored[:top_n]


# ── Embedding method (optional) ────────────────────────────────────────────

_EMBEDDING_MODEL = None

def _get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model — all-MiniLM is 80MB, very fast on CPU
            model_name = os.getenv(
                "LAZY_SKILLS_EMBEDDING_MODEL",
                "all-MiniLM-L6-v2",
            )
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
            logger.info("lazy-load-skills: loaded embedding model %s", model_name)
        except ImportError:
            logger.warning(
                "lazy-load-skills: sentence-transformers not installed. "
                "Install with: pip install sentence-transformers. "
                "Falling back to keyword method."
            )
            return None
        except Exception as exc:
            logger.warning(
                "lazy-load-skills: failed to load embedding model: %s. "
                "Falling back to keyword method.",
                exc,
            )
            return None
    return _EMBEDDING_MODEL


def _embedding_relevance(
    user_message: str,
    skills: List[Dict[str, Any]],
    top_n: int = 5,
    min_score: float = 0.15,
) -> List[Dict[str, Any]]:
    """Rank skills by cosine similarity of embeddings."""
    model = _get_embedding_model()
    if model is None:
        # Fall back to keyword method
        logger.info("lazy-load-skills: embedding unavailable, using keyword fallback")
        return _keyword_relevance(user_message, skills, top_n, min_score)

    import numpy as np

    # Embed the user message
    query_embedding = model.encode([user_message], convert_to_numpy=True)[0]
    query_norm = np.linalg.norm(query_embedding)

    # Embed skill descriptions
    skill_texts = [
        f"{s.get('name', '')}: {s.get('description', '')}"
        for s in skills
    ]
    skill_embeddings = model.encode(skill_texts, convert_to_numpy=True)

    # Compute cosine similarities
    scored = []
    for i, skill in enumerate(skills):
        skill_norm = np.linalg.norm(skill_embeddings[i])
        if query_norm == 0 or skill_norm == 0:
            similarity = 0.0
        else:
            similarity = float(
                np.dot(query_embedding, skill_embeddings[i])
                / (query_norm * skill_norm)
            )

        if similarity >= min_score:
            scored.append({
                "name": skill.get("name", ""),
                "description": skill.get("description", ""),
                "score": round(similarity, 4),
            })

    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored[:top_n]


# ── Public API ─────────────────────────────────────────────────────────────

def compute_relevance(
    user_message: str,
    skills: List[Dict[str, Any]],
    method: str = "keyword",
    top_n: int = 5,
    min_score: float = 0.15,
) -> List[Dict[str, Any]]:
    """Compute relevance scores for all skills against a user message.

    Args:
        user_message: The user's raw message text.
        skills: List of skill dicts with ``name`` and ``description`` keys.
        method: ``"keyword"`` or ``"embedding"``.
        top_n: Maximum number of skills to return.
        min_score: Minimum relevance score (0.0–1.0) to include a skill.

    Returns:
        List of skill dicts with ``name``, ``description``, and ``score``,
        sorted by relevance descending, limited to ``top_n``.
    """
    if method == "embedding":
        return _embedding_relevance(user_message, skills, top_n, min_score)
    return _keyword_relevance(user_message, skills, top_n, min_score)


def get_available_skills() -> List[Dict[str, Any]]:
    """Discover all available skills from the Hermes skills directory.

    Parses SKILL.md files for YAML frontmatter to extract name and
    description. Falls back to filename-based extraction if SKILL.md
    is unavailable.
    """
    skills = []

    # Find the skills directory
    skills_dir = _find_skills_dir()
    if not skills_dir:
        return skills

    try:
        import yaml
    except ImportError:
        yaml = None

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_md = skill_dir / "SKILL.md"
        name = skill_dir.name
        description = ""

        if skill_md.exists():
            try:
                content = skill_md.read_text(encoding="utf-8", errors="replace")
                # Extract YAML frontmatter between --- markers
                fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
                if fm_match and yaml:
                    try:
                        frontmatter = yaml.safe_load(fm_match.group(1))
                        if isinstance(frontmatter, dict):
                            name = frontmatter.get("name", name)
                            description = frontmatter.get("description", "")
                    except yaml.YAMLError:
                        pass
                elif fm_match:
                    # Crude frontmatter parse without yaml
                    fm = fm_match.group(1)
                    for line in fm.split("\n"):
                        line = line.strip()
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip().strip('"').strip("'")
                        elif line.startswith("description:"):
                            description = line.split(":", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass

        skills.append({
            "name": name,
            "description": description,
            "path": str(skill_dir),
        })

    return skills


def _find_skills_dir() -> Optional[Path]:
    """Locate the Hermes skills directory."""
    # Check env override first
    env_dir = os.getenv("HERMES_SKILLS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    # Default locations
    candidates = [
        Path.home() / ".hermes" / "skills",
        Path("/usr/local/lib/hermes-agent/skills"),
    ]

    # Also check HERMES_HOME
    hermes_home = os.getenv("HERMES_HOME")
    if hermes_home:
        candidates.insert(0, Path(hermes_home) / "skills")

    for c in candidates:
        if c.is_dir():
            return c

    return None
