"""Prompt Registry — versioned prompt management for all agents."""

import json
import logging
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Loads and renders versioned agent prompts from Markdown files."""

    _registry: dict = {}

    @classmethod
    def load(cls) -> None:
        registry_path = _PROMPT_DIR / "registry.json"
        if not registry_path.exists():
            logger.warning("Prompt registry file not found: %s", registry_path)
            cls._registry = {"prompts": {}}
            return
        try:
            cls._registry = json.loads(registry_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load prompt registry: %s", exc)
            cls._registry = {"prompts": {}}

    @classmethod
    def get(cls, prompt_id: str, **kwargs) -> str:
        """Get a rendered prompt by its registry ID."""
        entry = cls._registry.get("prompts", {}).get(prompt_id)
        if not entry:
            raise KeyError(f"Prompt '{prompt_id}' not found in registry")
        file_path = _PROMPT_DIR / entry["path"]
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        template = file_path.read_text()
        if kwargs:
            try:
                template = template.format(**kwargs)
            except (KeyError, ValueError, IndexError) as exc:
                logger.warning(
                    "Failed to render prompt '%s': %s. "
                    "Check template variables match the provided kwargs.",
                    prompt_id, exc,
                )
        return template
