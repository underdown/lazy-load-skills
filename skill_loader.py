"""
Skill content loader.

Loads the full SKILL.md content for a skill by name. Handles:
- User skills in ``~/.hermes/skills/<name>/SKILL.md``
- Nested skills from the bundled skills tree (recursive search)
- Graceful fallback if a skill file is missing or unreadable
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Cache: skill name → absolute path to SKILL.md
_path_cache: Dict[str, Path] = {}


def get_skills_dir() -> Optional[Path]:
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


def _build_path_cache() -> None:
    """Walk the skills tree and cache name→path mappings.

    Skills can be at any depth (e.g., devops/docker-management/SKILL.md).
    We match by the ``name`` field in YAML frontmatter, not directory name.
    """
    global _path_cache

    skills_dir = get_skills_dir()
    if not skills_dir:
        return

    try:
        import yaml
    except ImportError:
        yaml = None

    for root, dirs, files in os.walk(skills_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        skill_md = Path(root) / "SKILL.md"
        if not skill_md.exists():
            continue

        name = Path(root).name
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
            fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if fm_match and yaml:
                try:
                    frontmatter = yaml.safe_load(fm_match.group(1))
                    if isinstance(frontmatter, dict):
                        name = frontmatter.get("name", name)
                except yaml.YAMLError:
                    pass
            elif fm_match:
                for line in fm_match.group(1).split("\n"):
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

        _path_cache[name] = skill_md

    logger.debug(
        "lazy-load-skills: cached %d skill paths", len(_path_cache)
    )


def load_skill_content(name: str) -> str:
    """Load the full SKILL.md content for a named skill.

    Searches the entire skills tree recursively to find the SKILL.md
    whose frontmatter ``name`` field matches.

    Args:
        name: Skill name (from SKILL.md frontmatter ``name`` field).

    Returns:
        Full markdown content of the SKILL.md file (with frontmatter
        stripped), or empty string if the skill can't be found or read.
    """
    # Build cache on first call
    if not _path_cache:
        _build_path_cache()

    skill_md = _path_cache.get(name)
    if skill_md is None:
        logger.debug("lazy-load-skills: SKILL.md not found for '%s'", name)
        return ""

    if not skill_md.exists():
        logger.debug("lazy-load-skills: cached path gone for '%s'", name)
        return ""

    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        # Strip YAML frontmatter
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx != -1:
                content = content[end_idx + 3:].lstrip("\n")
        return content.strip()
    except Exception as exc:
        logger.warning(
            "lazy-load-skills: failed to read SKILL.md for '%s': %s",
            name, exc,
        )
        return ""
