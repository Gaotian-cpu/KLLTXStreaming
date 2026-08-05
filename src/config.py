from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import OmegaConf


def load_config(path: str | Path) -> OmegaConf:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    return cfg


def merge_cli_overrides(cfg: OmegaConf, overrides: Optional[dict[str, Any]] = None) -> OmegaConf:
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg
