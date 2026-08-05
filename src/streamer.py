from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image

from .utils import frames_to_pil, save_frames_as_video, setup_logger

logger = setup_logger("streamer")


class FrameStreamer:
    """
    负责把生成好的帧立即输出。
    支持：
      - progressive 写文件
      - 内存 Queue（供其他消费者）
      - 简单 WebSocket（可选）
    """

    def __init__(
        self,
        mode: str = "file",
        output_dir: str = "outputs",
        fps: int = 24,
        progressive_save: bool = True,
        ws_host: str = "0.0.0.0",
        ws_port: int = 8765,
    ):
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.progressive_save = progressive_save
        self.ws_host = ws_host
        self.ws_port = ws_port

        self._frame_queue: queue.Queue = queue.Queue(maxsize=256)
        self._all_frames: List[Image.Image] = []
        self._chunk_idx = 0
        self._session_id = time.strftime("%Y%m%d_%H%M%S")
        self._lock = threading.Lock()

        self._ws_server = None
        if mode == "websocket":
            self._start_ws_server()

    def push_frames(self, frames: List[Image.Image], chunk_idx: int) -> None:
        """推入一个 chunk 的帧，立即处理输出。"""
        frames = frames_to_pil(frames)
        with self._lock:
            self._all_frames.extend(frames)
            self._chunk_idx = chunk_idx

        for f in frames:
            try:
                self._frame_queue.put_nowait(f)
            except queue.Full:
                pass  # 消费者太慢就丢帧，保证生成端不堵

        if self.progressive_save and self.mode in ("file", "queue"):
            chunk_path = self.output_dir / f"{self._session_id}_chunk_{chunk_idx:04d}.mp4"
            try:
                save_frames_as_video(frames, chunk_path, fps=self.fps)
                logger.info(f"Chunk saved → {chunk_path}")
            except Exception as e:
                logger.error(f"Failed to save chunk: {e}")

        if self.mode == "websocket" and self._ws_server is not None:
            # 简单实现：只通知有新 chunk，实际可扩展为推送 JPEG
            logger.debug(f"WS notify chunk {chunk_idx}")

    def finalize(self, final_name: Optional[str] = None) -> Path:
        """把所有帧合成最终视频。"""
        final_name = final_name or f"{self._session_id}_full.mp4"
        final_path = self.output_dir / final_name
        with self._lock:
            if not self._all_frames:
                logger.warning("No frames to finalize.")
                return final_path
            save_frames_as_video(self._all_frames, final_path, fps=self.fps)
        logger.info(f"Final video → {final_path} ({len(self._all_frames)} frames)")
        return final_path

    def get_frame_queue(self) -> queue.Queue:
        return self._frame_queue

    def _start_ws_server(self) -> None:
        """可选的简易 WebSocket 服务（需要时再启用）。"""
        try:
            import asyncio
            import websockets

            async def handler(websocket):
                await websocket.send(f"LTX Stream connected. session={self._session_id}")
                while True:
                    await asyncio.sleep(1.0)

            def run():
                asyncio.set_event_loop(asyncio.new_event_loop())
                loop = asyncio.get_event_loop()
                start = websockets.serve(handler, self.ws_host, self.ws_port)
                loop.run_until_complete(start)
                loop.run_forever()

            t = threading.Thread(target=run, daemon=True)
            t.start()
            logger.info(f"WebSocket server on ws://{self.ws_host}:{self.ws_port}")
        except ImportError:
            logger.warning("websockets not installed, skip WS server.")
        except Exception as e:
            logger.error(f"WS server failed: {e}")
