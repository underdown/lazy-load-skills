"""
Skill content loader.

Loads the full SKILL.md content for a skill by name. Handles:
- User skills in ``~/.hermes/skills/<name>/SKILL.md``
- Bundled skills (plugin-provided) looked up through common paths
- Graceful fallback if a skill file is missing or unreadable
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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


def load_skill_content(name: str) -> str:
    """Load the full SKILL.md content for a named skill.

    Args:
        name: Skill name (directory name under skills/).

    Returns:
        Full markdown content of the SKILL.md file, or empty string if
        the skill can't be found or read.
    """
    skills_dir = get_skills_dir()
    if not skills_dir:
        logger.debug("lazy-load-skills: skills dir not found")
        return ""

    skill_md = skills_dir / name / "SKILL.md"
    if not skill_md.exists():
        logger.debug("lazy-load-skills: SKILL.md not found for %s", name)
        return ""

    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        # Strip YAML frontmatter (already parsed for metadata)
        if content.startswith("---"):
            # Find the closing ---
            end_idx = content.find("---", 3)
            if end_idx != -1:
                content = content[end_idx + 3:].lstrip("\n")
        return content.strip()
    except Exception as exc:
        logger.warning(
            "lazy-load-skills: failed to read SKILL.md for %s: %s",
            name, exc,
        )
        return ""
