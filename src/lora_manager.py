from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .utils import setup_logger

logger = setup_logger("lora_manager")


class LoRAManager:
    """
    动态 LoRA 管理器。
    在 chunk 边界热加载 / 卸载 LoRA。
    实际加载逻辑依赖官方 pipeline 的 API，这里提供统一接口。
    """

    def __init__(self, pipeline: Any = None, default_scale: float = 0.8):
        self.pipeline = pipeline
        self.default_scale = default_scale
        self._current_path: Optional[str] = None
        self._current_scale: float = default_scale
        self._pending: Optional[tuple[str, float]] = None  # (path, scale)
        self._unload_flag = False

    def attach_pipeline(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def request_load(self, path: str, scale: Optional[float] = None) -> None:
        """请求在下一个 chunk 边界加载 LoRA。"""
        path = str(Path(path).resolve())
        if not Path(path).exists():
            logger.error(f"LoRA file not found: {path}")
            return
        self._pending = (path, scale if scale is not None else self.default_scale)
        self._unload_flag = False
        logger.info(f"LoRA load requested: {path} (scale={self._pending[1]})")

    def request_unload(self) -> None:
        self._unload_flag = True
        self._pending = None
        logger.info("LoRA unload requested.")

    def apply_if_needed(self) -> None:
        """
        在 chunk 开始前调用，真正执行加载/卸载。
        这里做了兼容处理：优先使用 pipeline 的 load_lora_weights / unload 方法，
        如果不存在则记录警告（用户可按官方 API 扩展）。
        """
        if self.pipeline is None:
            return

        if self._unload_flag:
            self._do_unload()
            self._unload_flag = False
            return

        if self._pending is not None:
            path, scale = self._pending
            self._do_load(path, scale)
            self._pending = None

    def _do_load(self, path: str, scale: float) -> None:
        try:
            # 常见 Diffusers / 官方风格接口
            if hasattr(self.pipeline, "load_lora_weights"):
                self.pipeline.load_lora_weights(path)
                if hasattr(self.pipeline, "set_adapters"):
                    # 部分实现需要显式设置 scale
                    pass
                logger.info(f"LoRA loaded: {path} (scale={scale})")
            elif hasattr(self.pipeline, "load_lora"):
                self.pipeline.load_lora(path, scale=scale)
                logger.info(f"LoRA loaded via load_lora: {path}")
            else:
                logger.warning(
                    "Pipeline does not expose load_lora_weights / load_lora. "
                    "Please extend LoRAManager._do_load for your LTX version."
                )
            self._current_path = path
            self._current_scale = scale
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")

    def _do_unload(self) -> None:
        try:
            if hasattr(self.pipeline, "unload_lora_weights"):
                self.pipeline.unload_lora_weights()
                logger.info("LoRA unloaded.")
            elif hasattr(self.pipeline, "disable_lora"):
                self.pipeline.disable_lora()
                logger.info("LoRA disabled.")
            else:
                logger.warning("No unload method found on pipeline.")
            self._current_path = None
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")

    @property
    def current(self) -> Optional[str]:
        return self._current_path
