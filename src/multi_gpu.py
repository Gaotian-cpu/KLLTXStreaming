from __future__ import annotations

import os
from typing import Optional

import torch

from .utils import get_available_gpus, setup_logger

logger = setup_logger("multi_gpu")


def init_distributed() -> tuple[int, int, int]:
    """
    返回 (local_rank, world_size, global_rank)
    兼容 torchrun 和单进程。
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        global_rank = int(os.environ.get("RANK", 0))
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Distributed init: rank={global_rank}, world={world_size}, local={local_rank}")
        return local_rank, world_size, global_rank
    else:
        return 0, 1, 0


def is_main_process() -> bool:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def resolve_device_map(strategy: str = "auto", device_map: str = "auto") -> Optional[str | dict]:
    """
    根据策略返回传给 pipeline 的 device_map。
    H100/H200 单卡通常足够放下蒸馏模型，多卡时用 auto 做简单切分。
    """
    n_gpu = get_available_gpus()
    if strategy == "single" or n_gpu <= 1:
        return None  # 手动 .to("cuda")
    if strategy == "auto" or strategy == "ddp":
        return device_map  # "auto" 让 accelerate / 官方处理
    return None


def barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
