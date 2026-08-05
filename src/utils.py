from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image


def setup_logger(name: str = "ltx_stream", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def frames_to_pil(frames: Union[torch.Tensor, np.ndarray, List]) -> List[Image.Image]:
    """Convert various frame formats to list of PIL Images."""
    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], Image.Image):
        return frames

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float()
        if frames.ndim == 5:  # B T C H W
            frames = frames[0]
        if frames.ndim == 4 and frames.shape[1] in (1, 3):  # T C H W
            frames = frames.permute(0, 2, 3, 1)
        frames = frames.numpy()

    if isinstance(frames, np.ndarray):
        if frames.ndim == 4 and frames.shape[-1] not in (1, 3):
            # T C H W
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)
        return [Image.fromarray(f) for f in frames]

    raise TypeError(f"Unsupported frames type: {type(frames)}")


def save_frames_as_video(
    frames: List[Image.Image],
    path: str | Path,
    fps: int = 24,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio

    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for img in frames:
        writer.append_data(np.array(img))
    writer.close()


def get_available_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()
