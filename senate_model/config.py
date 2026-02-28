from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
_config: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, "r") as f:
            _config = json.load(f)
    return _config


def get(*keys: str, default: Any = None) -> Any:
    """Dot-path accessor, e.g. get('bills', 'max_provisions')"""
    node = load()
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node