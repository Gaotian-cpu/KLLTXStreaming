from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import OmegaConf
from pathlib import Path


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


def resolve_model_paths(cfg: OmegaConf, model_root: Optional[str] = None) -> OmegaConf:
    """
    将 cfg.model 下的相对路径与 model_root 拼接为绝对路径。
    若某项已是绝对路径，则不拼接。
    model_root 为空时：相对路径相对「当前工作目录」解析。
    """
    if model_root:
        root = Path(model_root).expanduser().resolve()
    else:
        root = Path.cwd()

    def _resolve(p) -> Optional[str]:
        if p is None:
            return None
        s = str(p).strip()
        if not s or s.lower() in ("null", "none"):
            return None
        path = Path(s).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        return str((root / path).resolve())

    # 确保 model 节点存在
    if not hasattr(cfg, "model") or cfg.model is None:
        return cfg

    keys = ("checkpoint", "spatial_upsampler", "text_encoder", "text_encoder_root")
    for key in keys:
        if key not in cfg.model:
            continue
        val = cfg.model.get(key)
        if val is None:
            continue
        cfg.model[key] = _resolve(val)

    return cfg
