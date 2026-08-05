from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

import torch
from omegaconf import OmegaConf
from PIL import Image

from .hls.writer import HLSWriter
from .lora_manager import LoRAManager
from .multi_gpu import barrier, init_distributed, is_main_process, resolve_device_map
from .prompt_manager import PromptManager
from .streamer import FrameStreamer
from .utils import frames_to_pil, get_available_gpus, seed_everything, setup_logger

logger = setup_logger("engine")


class StreamingEngine:
    """
    LTX 流式交互核心引擎。

    设计要点：
    - 每次生成一个短 chunk
    - 用上一 chunk 尾帧作为下一 chunk 的视觉条件（continuation）
    - 新提示词在 chunk 边界生效
    - 动态 LoRA 在 chunk 边界热加载
    - 支持单卡 / 多卡（device_map 或 torchrun）
    """

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.local_rank, self.world_size, self.global_rank = init_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        seed_everything(int(cfg.generation.seed) + self.global_rank)

        self.prompt_mgr = PromptManager(
            maxsize=int(cfg.runtime.prompt_queue_size),
            initial_prompt="",
        )
        self.lora_mgr = LoRAManager(default_scale=float(cfg.lora.scale))
        self.streamer = FrameStreamer(
            mode=cfg.streaming.mode,
            output_dir=cfg.streaming.output_dir,
            fps=int(cfg.generation.fps),
            progressive_save=bool(cfg.streaming.progressive_save),
            ws_host=cfg.streaming.ws_host,
            ws_port=int(cfg.streaming.ws_port),
        )

        self.pipeline: Any = None
        self._last_frames: List[Image.Image] = []
        self._chunk_count = 0
        self._running = False

        self._load_pipeline()

    # ------------------------------------------------------------------
    # Pipeline 加载（适配层）
    # ------------------------------------------------------------------
    def _load_pipeline(self) -> None:
        """
        优先尝试官方 ltx_pipelines.DistilledPipeline，
        失败则回退到 Diffusers LTXPipeline / LTXImageToVideoPipeline。
        你可以根据实际安装的版本修改这里。
        """
        device_map = resolve_device_map(
            strategy=self.cfg.multi_gpu.strategy,
            device_map=self.cfg.multi_gpu.device_map,
        )
        dtype = getattr(torch, self.cfg.model.dtype, torch.bfloat16)

        ckpt = self.cfg.model.get("checkpoint")
        spatial = self.cfg.model.get("spatial_upsampler")
        text_encoder = self.cfg.model.get("text_encoder")

        if not ckpt:
            raise RuntimeError("cfg.model.checkpoint 为空，请检查配置与 --model-root")

        if spatial:
            logger.info(f"spatial_upsampler={spatial}")
        if text_encoder:
            logger.info(f"text_encoder(file)={text_encoder}")

        # ---- 尝试官方 LTX-2 DistilledPipeline ----
        try:
            from ltx_pipelines import DistilledPipeline  # type: ignore

            logger.info("Loading official DistilledPipeline...")
            kwargs = {
                "torch_dtype": dtype,
            }
            if device_map is not None:
                kwargs["device_map"] = device_map

            # 实际参数名请对照你安装的 LTX-2 版本
            self.pipeline = DistilledPipeline.from_pretrained(
                ckpt,
                **kwargs,
            )
            if device_map is None:
                self.pipeline.to(self.device)
            logger.info("Official DistilledPipeline loaded.")
        except Exception as e:
            logger.warning(f"Official DistilledPipeline not available ({e}), trying Diffusers...")
            self._load_diffusers_fallback(dtype, device_map)

        self.lora_mgr.attach_pipeline(self.pipeline)

        # 初始 LoRA
        if self.cfg.lora.initial:
            self.lora_mgr.request_load(self.cfg.lora.initial, self.cfg.lora.scale)
            self.lora_mgr.apply_if_needed()

    def _load_diffusers_fallback(self, dtype, device_map) -> None:
        try:
            from diffusers import LTXPipeline, LTXImageToVideoPipeline  # type: ignore

            # 根据你下载的权重选择
            model_id = self.cfg.model.checkpoint
            logger.info(f"Loading Diffusers pipeline from {model_id}")
            try:
                self.pipeline = LTXImageToVideoPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device_map,
                )
            except Exception:
                self.pipeline = LTXPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device_map,
                )
            if device_map is None:
                self.pipeline.to(self.device)
            logger.info("Diffusers LTX pipeline loaded.")
        except Exception as e:
            logger.error(f"Failed to load any LTX pipeline: {e}")
            raise RuntimeError(
                "无法加载 LTX 模型。请确认已安装官方 LTX-2 或 Diffusers，并正确配置 checkpoint 路径。"
            ) from e

    # ------------------------------------------------------------------
    # 单个 Chunk 生成
    # ------------------------------------------------------------------
    def _generate_chunk(
        self,
        prompt: str,
        condition_image: Optional[Image.Image] = None,
        num_frames: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        调用底层 pipeline 生成一个 chunk。
        这里做了兼容处理，实际参数名请按你的 LTX 版本微调。
        """
        num_frames = num_frames or int(self.cfg.generation.chunk_frames)
        width = int(self.cfg.generation.width)
        height = int(self.cfg.generation.height)
        steps = int(self.cfg.generation.num_inference_steps)
        guidance = float(self.cfg.generation.guidance_scale)
        neg = self.cfg.generation.negative_prompt

        # 动态 LoRA 在 chunk 边界生效
        self.lora_mgr.apply_if_needed()

        kwargs = {
            "prompt": prompt,
            "negative_prompt": neg,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }

        # 续写条件
        if condition_image is not None:
            # 不同 pipeline 参数名可能是 image / conditioning_image / first_frame 等
            if hasattr(self.pipeline, "__call__"):
                # 尝试常见参数
                for key in ("image", "conditioning_image", "first_frame", "condition_image"):
                    kwargs[key] = condition_image
                    break

        t0 = time.time()
        try:
            with torch.inference_mode():
                output = self.pipeline(**kwargs)
        except TypeError:
            # 参数名不匹配时的兜底：去掉 image 相关再试
            kwargs.pop("image", None)
            kwargs.pop("conditioning_image", None)
            kwargs.pop("first_frame", None)
            kwargs.pop("condition_image", None)
            with torch.inference_mode():
                output = self.pipeline(**kwargs)

        # 解析输出
        if hasattr(output, "frames"):
            frames = output.frames
            if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
                frames = frames[0]
        elif hasattr(output, "images"):
            frames = output.images
        else:
            frames = output

        frames = frames_to_pil(frames)
        elapsed = time.time() - t0
        logger.info(
            f"Chunk {self._chunk_count} generated: {len(frames)} frames, "
            f"{elapsed:.2f}s ({len(frames)/max(elapsed,1e-3):.1f} fps)"
        )
        return frames

    # ------------------------------------------------------------------
    # 主循环：流式生成
    # ------------------------------------------------------------------
    def run(
        self,
        initial_prompt: str,
        initial_image: Optional[Image.Image] = None,
        max_chunks: Optional[int] = None,
    ) -> Path:
        """
        开始流式生成。
        可在其他线程中调用 self.prompt_mgr.update(...) 动态改提示词。
        """
        max_chunks = max_chunks or int(self.cfg.runtime.max_chunks)
        self.prompt_mgr.update(initial_prompt)
        self._running = True
        self._chunk_count = 0
        self._last_frames = []

        condition = initial_image
        overlap = int(self.cfg.generation.overlap_frames)

        logger.info("=" * 60)
        logger.info("Streaming started")
        logger.info(f"GPUs: {get_available_gpus()}, world_size={self.world_size}")
        logger.info(f"Chunk frames={self.cfg.generation.chunk_frames}, overlap={overlap}")
        logger.info("=" * 60)

        try:
            while self._running and self._chunk_count < max_chunks:
                if self.prompt_mgr.stopped:
                    logger.info("Stop signal received.")
                    break

                # 获取最新提示词（可能刚被用户更新）
                prompt = self.prompt_mgr.consume_latest()
                if not prompt:
                    time.sleep(0.1)
                    continue

                # 生成当前 chunk
                frames = self._generate_chunk(
                    prompt=prompt,
                    condition_image=condition,
                )

                if not frames:
                    logger.warning("Empty chunk, skipping.")
                    continue

                # 立即流式输出
                if is_main_process():
                    self.streamer.push_frames(frames, self._chunk_count)

                # 准备下一 chunk 的条件（取尾部 overlap 帧的最后一帧）
                if overlap > 0 and len(frames) >= overlap:
                    condition = frames[-1]  # 也可以用中间帧做更平滑过渡
                else:
                    condition = frames[-1] if frames else condition

                self._last_frames = frames
                self._chunk_count += 1
                barrier()

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._running = False

        final_path = Path(".")
        if is_main_process():
            final_path = self.streamer.finalize()
        barrier()
        return final_path

    def stop(self) -> None:
        self._running = False
        self.prompt_mgr.stop()

    # 便捷接口
    def update_prompt(self, text: str) -> None:
        self.prompt_mgr.update(text)

    def load_lora(self, path: str, scale: float = 0.8) -> None:
        self.lora_mgr.request_load(path, scale)

    def unload_lora(self) -> None:
        self.lora_mgr.request_unload()

    # ------------------------------------------------------------------
    # 任务模式：生成 HLS (m3u8) 流，供 HTTP 服务调用
    # ------------------------------------------------------------------
    def run_task(self, task_info, task_manager) -> None:
        """
        由 TaskManager worker 调用。
        生成过程中持续写入 m3u8 分片，状态变为 streaming 后浏览器即可播放。
        """
        from .task_manager import TaskStatus

        task_id = task_info.task_id
        prompt = task_info.prompt
        max_chunks = int(task_info.max_chunks)
        segment_duration = float(task_info.segment_duration)
        fps = int(self.cfg.generation.fps)

        # 按分片时长计算每块帧数
        chunk_frames = max(8, int(round(segment_duration * fps)))
        # 临时覆盖配置
        self.cfg.generation.chunk_frames = chunk_frames

        task_dir = Path(task_manager.streams_root) / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        hls = HLSWriter(
            output_dir=task_dir,
            segment_duration=segment_duration,
            fps=fps,
            playlist_name="playlist.m3u8",
        )

        # 流地址（相对路径，API 层拼 base_url）
        stream_rel = f"streams/{task_id}/playlist.m3u8"

        condition = None
        if task_info.image_path:
            try:
                condition = Image.open(task_info.image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Load input image failed: {e}")

        # 可选覆盖分辨率
        if task_info.width:
            self.cfg.generation.width = int(task_info.width)
        if task_info.height:
            self.cfg.generation.height = int(task_info.height)

        self._running = True
        self._chunk_count = 0
        overlap = int(self.cfg.generation.overlap_frames)

        logger.info(f"[Task {task_id}] start HLS generation, segment={segment_duration}s, chunks={max_chunks}")

        try:
            for i in range(max_chunks):
                if not self._running or task_info.status == TaskStatus.CANCELLED:
                    break

                # 支持中途改提示词（若有外部 update）
                current_prompt = self.prompt_mgr.consume_latest() or prompt
                if i == 0:
                    current_prompt = prompt

                frames = self._generate_chunk(
                    prompt=current_prompt,
                    condition_image=condition,
                    num_frames=chunk_frames,
                )
                if not frames:
                    continue

                seg = hls.add_frames(frames)
                self._chunk_count += 1
                progress = min(1.0, self._chunk_count / max(max_chunks, 1))

                if seg is not None:
                    # 第一个分片写好后就可以播放
                    status = TaskStatus.STREAMING if self._chunk_count >= 1 else TaskStatus.RUNNING
                    task_manager.update_status(
                        task_id,
                        status=status,
                        stream_url=stream_rel,
                        segment_count=hls.segment_count,
                        progress=progress,
                        playlist_path=str(hls.playlist_path),
                    )

                # 续写条件
                condition = frames[-1] if frames else condition

            hls.finalize()
            task_manager.update_status(
                task_id,
                status=TaskStatus.COMPLETED,
                stream_url=stream_rel,
                segment_count=hls.segment_count,
                progress=1.0,
                playlist_path=str(hls.playlist_path),
            )
            logger.info(f"[Task {task_id}] completed, segments={hls.segment_count}")
        except Exception as e:
            logger.exception(f"[Task {task_id}] error: {e}")
            try:
                hls.finalize()
            except Exception:
                pass
            task_manager.update_status(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                stream_url=stream_rel if hls.segment_count > 0 else None,
                segment_count=hls.segment_count,
            )
        finally:
            self._running = False
