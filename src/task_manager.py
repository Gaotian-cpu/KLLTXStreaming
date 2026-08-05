from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .utils import setup_logger

logger = setup_logger("task_manager")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    STREAMING = "streaming"   # 已有可播放分片
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    task_id: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # 可选初始图片路径（服务端保存后的路径）
    image_path: Optional[str] = None
    # 用户可选参数
    segment_duration: float = 3.0
    max_chunks: int = 20
    width: Optional[int] = None
    height: Optional[int] = None
    # 运行时信息
    stream_url: Optional[str] = None          # 相对或绝对 m3u8 地址
    playlist_path: Optional[str] = None
    segment_count: int = 0
    error: Optional[str] = None
    progress: float = 0.0                     # 0~1
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, base_url: str = "") -> dict:
        d = {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "segment_duration": self.segment_duration,
            "max_chunks": self.max_chunks,
            "segment_count": self.segment_count,
            "progress": round(self.progress, 3),
            "error": self.error,
        }
        if self.stream_url:
            # 返回可直接给浏览器用的地址
            if self.stream_url.startswith("http"):
                d["stream_url"] = self.stream_url
            else:
                d["stream_url"] = f"{base_url.rstrip('/')}/{self.stream_url.lstrip('/')}"
        else:
            d["stream_url"] = None
        return d


class TaskManager:
    """
    内存任务表 + 工作线程调度。
    生产环境可换成 Redis / DB。
    """

    def __init__(self, streams_root: str | Path, max_workers: int = 1):
        self.streams_root = Path(streams_root)
        self.streams_root.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.RLock()
        self._queue: List[str] = []          # task_id 队列
        self._workers: List[threading.Thread] = []
        self._stop = threading.Event()
        self._engine_factory = None          # 由外部注入：() -> StreamingEngine

    def set_engine_factory(self, factory) -> None:
        """注入创建 StreamingEngine 的工厂函数。"""
        self._engine_factory = factory

    def start_workers(self) -> None:
        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        logger.info(f"TaskManager started with {self.max_workers} worker(s)")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def submit(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        segment_duration: float = 3.0,
        max_chunks: int = 20,
        width: Optional[int] = None,
        height: Optional[int] = None,
        meta: Optional[dict] = None,
    ) -> TaskInfo:
        task_id = uuid.uuid4().hex[:16]
        task_dir = self.streams_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        image_path = None
        if image is not None:
            image_path = str(task_dir / "input.png")
            image.save(image_path)

        info = TaskInfo(
            task_id=task_id,
            prompt=prompt,
            image_path=image_path,
            segment_duration=segment_duration,
            max_chunks=max_chunks,
            width=width,
            height=height,
            meta=meta or {},
        )
        with self._lock:
            self._tasks[task_id] = info
            self._queue.append(task_id)
        logger.info(f"Task submitted: {task_id}")
        return info

    def get(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 50) -> List[TaskInfo]:
        with self._lock:
            items = list(self._tasks.values())
        items.sort(key=lambda x: x.created_at, reverse=True)
        return items[:limit]

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            info = self._tasks.get(task_id)
            if not info:
                return False
            if info.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False
            info.status = TaskStatus.CANCELLED
            info.updated_at = time.time()
            return True

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        stream_url: Optional[str] = None,
        segment_count: Optional[int] = None,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        playlist_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            info = self._tasks.get(task_id)
            if not info:
                return
            info.status = status
            info.updated_at = time.time()
            if stream_url is not None:
                info.stream_url = stream_url
            if segment_count is not None:
                info.segment_count = segment_count
            if progress is not None:
                info.progress = progress
            if error is not None:
                info.error = error
            if playlist_path is not None:
                info.playlist_path = playlist_path

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            task_id = None
            with self._lock:
                if self._queue:
                    task_id = self._queue.pop(0)
            if task_id is None:
                time.sleep(0.2)
                continue
            try:
                self._run_task(task_id)
            except Exception as e:
                logger.exception(f"Task {task_id} failed: {e}")
                self.update_status(task_id, TaskStatus.FAILED, error=str(e))

    def _run_task(self, task_id: str) -> None:
        info = self.get(task_id)
        if not info or info.status == TaskStatus.CANCELLED:
            return
        if self._engine_factory is None:
            self.update_status(task_id, TaskStatus.FAILED, error="engine_factory not set")
            return

        self.update_status(task_id, TaskStatus.RUNNING)
        engine = self._engine_factory()
        # 把任务参数传给引擎（通过闭包或属性）
        engine.run_task(info, self)
