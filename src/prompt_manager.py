from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .utils import setup_logger

logger = setup_logger("prompt_manager")


@dataclass
class PromptCommand:
    text: str
    timestamp: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


class PromptManager:
    """
    异步提示词管理器。
    支持在生成过程中随时推入新指令，下一个 chunk 会使用最新提示词。
    """

    def __init__(self, maxsize: int = 8, initial_prompt: str = ""):
        self._queue: queue.Queue[PromptCommand] = queue.Queue(maxsize=maxsize)
        self._current: str = initial_prompt
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def current(self) -> str:
        with self._lock:
            return self._current

    def update(self, text: str, **meta) -> None:
        """推入新提示词（线程安全）。"""
        cmd = PromptCommand(text=text.strip(), meta=meta)
        try:
            self._queue.put_nowait(cmd)
            logger.info(f"New prompt queued: {text[:80]}...")
        except queue.Full:
            # 丢弃最旧的，保留最新
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(cmd)
            logger.warning("Prompt queue full, dropped oldest.")

    def consume_latest(self) -> str:
        """
        取出队列中最新的提示词并设为 current。
        如果队列为空，返回当前提示词。
        """
        latest: Optional[PromptCommand] = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            with self._lock:
                self._current = latest.text
            logger.info(f"Active prompt switched → {latest.text[:80]}...")
        return self.current

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()
