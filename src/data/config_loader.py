"""Load configuration from config/settings.yaml and config/secrets.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning {} if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_settings() -> dict[str, Any]:
    """Load settings.yaml from config/."""
    return load_yaml(Path("config/settings.yaml"))


def load_secrets() -> dict[str, Any]:
    """Load secrets.yaml from config/. Returns {} if not configured."""
    return load_yaml(Path("config/secrets.yaml"))
