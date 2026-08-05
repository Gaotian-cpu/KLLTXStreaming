from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from ..utils import setup_logger

logger = setup_logger("hls_writer")


class HLSWriter:
    """
    将生成的帧写成 HLS (m3u8 + .ts) 分段。
    支持边生成边追加 segment，浏览器可边下边播。
    """

    def __init__(
        self,
        output_dir: str | Path,
        segment_duration: float = 3.0,
        fps: int = 24,
        playlist_name: str = "playlist.m3u8",
        segment_prefix: str = "seg",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_duration = float(segment_duration)
        self.fps = int(fps)
        self.playlist_name = playlist_name
        self.segment_prefix = segment_prefix

        self._segment_index = 0
        self._segment_durations: List[float] = []
        self._finished = False
        self._target_duration = max(1, int(round(self.segment_duration)) + 1)

        # 初始化空 playlist（直播风格，后续可改成 VOD）
        self._write_playlist(end=False)

    @property
    def playlist_path(self) -> Path:
        return self.output_dir / self.playlist_name

    @property
    def segment_count(self) -> int:
        return self._segment_index

    def add_frames(self, frames: List[Image.Image]) -> Optional[Path]:
        """
        把一帧序列写成一个 .ts 分片，并更新 m3u8。
        返回新分片路径；若失败返回 None。
        """
        if not frames or self._finished:
            return None

        # 实际时长按帧数计算
        duration = len(frames) / max(self.fps, 1)
        seg_name = f"{self.segment_prefix}_{self._segment_index:05d}.ts"
        seg_path = self.output_dir / seg_name

        ok = self._encode_ts(frames, seg_path)
        if not ok:
            return None

        self._segment_durations.append(duration)
        self._segment_index += 1
        self._target_duration = max(self._target_duration, int(round(duration)) + 1)
        self._write_playlist(end=False)
        logger.info(f"HLS segment written: {seg_name} ({duration:.2f}s)")
        return seg_path

    def finalize(self) -> Path:
        """结束流，写入 #EXT-X-ENDLIST。"""
        self._finished = True
        self._write_playlist(end=True)
        logger.info(f"HLS playlist finalized: {self.playlist_path}")
        return self.playlist_path

    def _write_playlist(self, end: bool = False) -> None:
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{self._target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:EVENT",  # 允许边生成边播
        ]
        for i, dur in enumerate(self._segment_durations):
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(f"{self.segment_prefix}_{i:05d}.ts")
        if end:
            lines.append("#EXT-X-ENDLIST")

        content = "\n".join(lines) + "\n"
        tmp = self.playlist_path.with_suffix(".m3u8.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.playlist_path)

    def _encode_ts(self, frames: List[Image.Image], out_path: Path) -> bool:
        """
        用 ffmpeg 把帧序列编码成 MPEG-TS。
        优先走管道，避免大量临时文件。
        """
        if not frames:
            return False

        w, h = frames[0].size
        # 保证偶数分辨率（H.264 要求）
        w = w - (w % 2)
        h = h - (h % 2)

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-g", str(max(1, int(self.fps * self.segment_duration))),  # keyframe 间隔 ≈ 分片时长
            "-f", "mpegts",
            str(out_path),
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert proc.stdin is not None
            for img in frames:
                if img.size != (w, h):
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
                proc.stdin.write(arr.tobytes())
            proc.stdin.close()
            ret = proc.wait(timeout=120)
            if ret != 0:
                err = proc.stderr.read().decode("utf-8", errors="ignore")[-500:]
                logger.error(f"ffmpeg failed (code={ret}): {err}")
                return False
            return out_path.exists() and out_path.stat().st_size > 0
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
            return False
        except Exception as e:
            logger.error(f"encode_ts error: {e}")
            return False
