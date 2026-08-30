"""Resolve the API key for the Tier 3 extraction model.

Shared by the ``evaluate_game.py`` and ``evaluate_batch.py`` CLIs so that
selecting an Anthropic/Claude extraction model automatically picks up the
Anthropic credentials, mirroring how ``run_game.py`` handles Claude agents.
"""

from __future__ import annotations

import os
from pathlib import Path

from quack.evaluation.tier3_statement_verification import (
    is_anthropic_extraction_model,
)


def resolve_extraction_api_key(model: str, explicit_key: str, root: Path) -> str:
    """Resolve the API key for the Tier 3 extraction model.

    Resolution order:

    * An explicit ``--api-key`` (``explicit_key``) always wins.
    * For Anthropic extraction models (e.g. ``claude-opus-4-7``): the key
      comes from ``$ANTHROPIC_API_KEY`` or ``claude_key.txt`` under ``root``
      (mirroring ``run_game.py``'s Claude-agent handling).
    * For OpenAI-compatible models (e.g. ``gpt-5.5``): the key comes from
      ``$QUACK_API_KEY`` or ``api_key.txt`` under ``root``.

    Returns an empty string (with a warning printed) if no key can be found.
    """
    if explicit_key:
        return explicit_key

    if is_anthropic_extraction_model(model):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
        key_path = root / "claude_key.txt"
        if key_path.exists():
            print(f"Loaded Anthropic API key from {key_path}")
            return key_path.read_text().strip()
        print(
            "WARNING: extraction model looks like Anthropic/Claude but no key "
            "found. Provide --api-key, set $ANTHROPIC_API_KEY, or create "
            "claude_key.txt"
        )
        return ""

    key = os.environ.get("QUACK_API_KEY", "")
    if key:
        return key
    key_path = root / "api_key.txt"
    if key_path.exists():
        print(f"Loaded API key from {key_path}")
        return key_path.read_text().strip()
    print("WARNING: --tier3 requires an API key. Provide --api-key or create api_key.txt")
    return ""
