"""
Skill relevance computation.

Supports two methods:
- ``keyword`` (default): TF-IDF-inspired keyword overlap with skill-name
  boosting. No dependencies. Sub-millisecond per query.
- ``embedding``: Uses sentence-transformers to compute cosine similarity
  between the user message and skill descriptions. More accurate for
  ambiguous queries. Requires ``pip install sentence-transformers``.

Scoring:
  Each skill gets a score 0.0–1.0. Skills below ``min_score`` are filtered.
  The top ``top_n`` skills are returned, ordered by relevance.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ── TF-IDF style keyword scoring ───────────────────────────────────────────

def _build_idf(skills: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute inverse document frequency for tokens across all skills.

    Tokens that appear in many skill descriptions get low IDF weights.
    Tokens that appear in few get high weights — these are the
    discriminative terms.
    """
    N = len(skills)
    if N == 0:
        return {}

    # Count how many skills each token appears in
    doc_freq: Counter = Counter()
    for skill in skills:
        name = skill.get("name", "")
        desc = skill.get("description", "")
        text = f"{name} {desc}"
        tokens = set(_tokenize(text))
        for t in tokens:
            doc_freq[t] += 1

    # Compute IDF: log(N / df) with smoothing
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((N + 1) / (df + 1)) + 1.0

    return idf


def _keyword_relevance(
    user_message: str,
    skills: List[Dict[str, Any]],
    top_n: int = 5,
    min_score: float = 0.15,
) -> List[Dict[str, Any]]:
    """Rank skills by TF-IDF-inspired keyword relevance.

    Scoring components (all 0.0–1.0):
    - idf_score (0.50): Weighted overlap using IDF — rare matching terms
      count more than common ones.
    - name_score (0.35): Direct hits in the skill NAME — heavily boosted
      because the name is a much stronger signal than description text.
    - coverage (0.15): What fraction of query tokens found a match?
      Prevents single-token matches from dominating.
    """
    query_tokens = _tokenize(user_message)
    if not query_tokens:
        return []

    # Build IDF weights
    idf = _build_idf(skills)
    if not idf:
        return []

    query_tf: Counter = Counter(query_tokens)

    scored = []
    for skill in skills:
        name = skill.get("name", "")
        desc = skill.get("description", "")

        # Split name into hyphenated/underscored parts for better matching
        name_parts = set()
        for part in re.split(r"[-_/]", name.lower()):
            part = part.strip()
            if part and part not in _STOP_WORDS:
                name_parts.add(part)
        # Also add the full tokenized name
        name_tokens = set(_tokenize(name))
        desc_tokens = set(_tokenize(desc))
        all_skill_tokens = name_tokens | desc_tokens | name_parts

        # ── IDF-weighted overlap ────────────────────────────────────
        idf_sum = 0.0
        max_possible_idf = 0.0
        for qt, qf in query_tf.items():
            qt_weight = idf.get(qt, 1.0)
            max_possible_idf += qf * qt_weight
            if qt in all_skill_tokens:
                idf_sum += qf * qt_weight

        idf_score = idf_sum / max(max_possible_idf, 0.001)

        # ── Name match score ────────────────────────────────────────
        # Direct name part matches get high weight
        name_hits = 0
        for qt in set(query_tokens):
            if qt in name_parts or qt in name_tokens:
                name_hits += 1
        name_score = min(name_hits / max(len(set(query_tokens)), 1), 1.0)

        # ── Query coverage ──────────────────────────────────────────
        unique_query = set(query_tokens)
        matched_query = unique_query & all_skill_tokens
        coverage = len(matched_query) / max(len(unique_query), 1)

        # ── Composite score ─────────────────────────────────────────
        score = (idf_score * 0.50) + (name_score * 0.35) + (coverage * 0.15)

        if score >= min_score:
            scored.append({
                "name": name,
                "description": desc,
                "score": round(min(score, 1.0), 4),
            })

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
        logger.info("lazy-load-skills: embedding unavailable, using keyword fallback")
        return _keyword_relevance(user_message, skills, top_n, min_score)

    import numpy as np

    query_embedding = model.encode([user_message], convert_to_numpy=True)[0]
    query_norm = np.linalg.norm(query_embedding)

    skill_texts = [
        f"{s.get('name', '')}: {s.get('description', '')}"
        for s in skills
    ]
    skill_embeddings = model.encode(skill_texts, convert_to_numpy=True)

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

    Walks the skills directory tree recursively. For each SKILL.md found,
    extracts the ``name`` and ``description`` from YAML frontmatter.

    Returns:
        List of skill dicts with keys: ``name``, ``description``, ``path``.
        Filtered to only return leaf skills (SKILL.md files), not empty
        category directories.
    """
    skills = []

    skills_dir = _find_skills_dir()
    if not skills_dir:
        return skills

    try:
        import yaml
    except ImportError:
        yaml = None

    # Walk recursively — skills can be nested 2-3 levels deep
    # (e.g., devops/docker-management/SKILL.md)
    for root, dirs, files in os.walk(skills_dir):
        # Skip dot-prefixed hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        skill_md = Path(root) / "SKILL.md"
        if not skill_md.exists():
            continue

        # Build relative path from skills_dir for name resolution
        rel_path = Path(root).relative_to(skills_dir)
        dir_name = rel_path.name if rel_path != Path(".") else skills_dir.name

        name = dir_name
        description = ""

        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
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
                for line in fm_match.group(1).split("\n"):
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

        # Skip umbrella SKILL.md files that are just category headers
        # (they typically have no tools or hooks in frontmatter)
        if not description and name == dir_name:
            continue

        skills.append({
            "name": name,
            "description": description,
            "path": str(Path(root)),
        })

    return skills


def _find_skills_dir() -> Optional[Path]:
    """Locate the Hermes skills directory."""
    env_dir = os.getenv("HERMES_SKILLS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    candidates = [
        Path.home() / ".hermes" / "skills",
        Path("/usr/local/lib/hermes-agent/skills"),
    ]

    hermes_home = os.getenv("HERMES_HOME")
    if hermes_home:
        candidates.insert(0, Path(hermes_home) / "skills")

    for c in candidates:
        if c.is_dir():
            return c

    return None
