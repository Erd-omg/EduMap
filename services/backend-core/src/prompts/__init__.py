"""Prompt Registry — versioned prompt management for all agents."""

import json
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


class PromptRegistry:
    """Loads and renders versioned agent prompts from Markdown files."""

    _registry: dict = {}

    @classmethod
    def load(cls) -> None:
        registry_path = _PROMPT_DIR / "registry.json"
        if registry_path.exists():
            cls._registry = json.loads(registry_path.read_text())

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
            template = template.format(**kwargs)
        return template


# Auto-load on import
PromptRegistry.load()
