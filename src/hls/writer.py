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

    def add_frames(self, frames: List[Image.Image],
                   audio_np: Optional["np.ndarray"] = None,  # shape [C, T] float32，约 [-1, 1]
                   sample_rate: int = 24000,) -> Optional[Path]:
        """
        把一帧序列写成一个 .ts 分片，并更新 m3u8。
        返回新分片路径；若失败返回 None。
        """
        if not frames or self._finished:
            return None

        duration = len(frames) / max(self.fps, 1)
        seg_name = f"{self.segment_prefix}_{self._segment_index:05d}.ts"
        seg_path = self.output_dir / seg_name

        ok = self._encode_ts(
            frames,
            seg_path,
            audio_np=audio_np,
            sample_rate=sample_rate,
        )
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

    def _encode_ts(
            self,
            frames: List[Image.Image],
            seg_path: Path,
            audio_np: Optional[np.ndarray] = None,
            sample_rate: int = 24000,
    ) -> bool:
        """将 frames（+ 可选音频）编码为 MPEG-TS。"""
        import subprocess
        import tempfile
        import shutil

        work = Path(tempfile.mkdtemp(prefix="hls_seg_"))
        try:
            # 1) 写 PNG 序列
            for i, im in enumerate(frames):
                im.convert("RGB").save(work / f"frame_{i:05d}.png")

            cmd = [
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-framerate", str(self.fps),
                "-i", str(work / "frame_%05d.png"),
            ]

            # 2) 可选音频：写成临时 wav 再接上
            wav_path = None
            if audio_np is not None and audio_np.size > 0:
                wav_path = work / "audio.wav"
                self._write_wav(wav_path, audio_np, sample_rate)
                cmd += ["-i", str(wav_path)]

            cmd += [
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-g", str(max(self.fps, 1)),          # 关键帧间隔约 1s，便于 HLS
                "-sc_threshold", "0",
            ]

            if wav_path is not None:
                cmd += [
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-ar", "48000",                    # 播放端常见采样率；也可保持 sample_rate
                    "-ac", "2",
                    "-shortest",                      # 以较短的一路为准，避免尾部空音/空画
                ]
            else:
                cmd += ["-an"]

            cmd += [
                "-f", "mpegts",
                str(seg_path),
            ]

            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                logger.error("ffmpeg failed: %s", r.stderr[-2000:] if r.stderr else "")
                return False
            return seg_path.is_file() and seg_path.stat().st_size > 0
        except Exception as e:
            logger.exception("encode_ts error: %s", e)
            return False
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _write_wav(self, path: Path, audio_np: np.ndarray, sample_rate: int) -> None:
        """audio_np: [C, T] 或 [T]，float，优先按 [-1,1] 解释。"""
        import wave

        x = np.asarray(audio_np, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]          # [1, T]
        if x.ndim != 2:
            raise ValueError(f"audio_np shape invalid: {x.shape}")

        # [C, T] → [T, C]
        x = np.transpose(x, (1, 0))
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.5:                    # 已是较大数值时做简单归一
            x = x / peak
        x = np.clip(x, -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16)
        channels = pcm.shape[1]

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())
