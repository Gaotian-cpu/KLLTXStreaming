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
        from pathlib import Path
        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.types import OffloadMode

        ckpt = self.cfg.model.get("checkpoint")
        spatial = self.cfg.model.get("spatial_upsampler")
        gemma_root = (
                self.cfg.model.get("text_encoder_root")
                or self.cfg.model.get("gemma_root")
        )

        if not ckpt or not Path(ckpt).is_file():
            raise FileNotFoundError(f"checkpoint 不是有效文件: {ckpt}")
        if not spatial or not Path(spatial).is_file():
            raise FileNotFoundError(f"spatial_upsampler 不是有效文件: {spatial}")
        if not gemma_root or not Path(gemma_root).is_dir():
            raise FileNotFoundError(
                f"gemma_root 必须是目录（官方 PromptEncoder 需要）: {gemma_root}"
            )

        logger.info(
            "Loading DistilledPipeline ckpt=%s spatial=%s gemma_root=%s",
            ckpt, spatial, gemma_root,
        )

        quant = build_fp8_cast_policy(str(ckpt))

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=str(ckpt),
            gemma_root=str(gemma_root),
            spatial_upsampler_path=str(spatial),
            loras=(),  # Fixme 增加lora
            device=self.device if str(self.device).startswith("cuda") else None,
            offload_mode=OffloadMode.CPU,
            quantization=quant,
        )

        # 推理不需要梯度
        def _disable_grad(module):
            if module is None:
                return
            if hasattr(module, "eval"):
                module.eval()
            for p in getattr(module, "parameters", lambda: [])():
                p.requires_grad_(False)

        for name in ("prompt_encoder", "image_conditioner", "stage", "upsampler", "video_decoder", "audio_decoder"):
            _disable_grad(getattr(self.pipeline, name, None))

        self.lora_mgr.attach_pipeline(self.pipeline)
        logger.info("DistilledPipeline loaded.")

    # def _load_diffusers_fallback(self, dtype, device_map) -> None:
    #     try:
    #         from diffusers import LTXPipeline, LTXImageToVideoPipeline  # type: ignore
    #
    #         # 根据你下载的权重选择
    #         model_id = self.cfg.model.checkpoint
    #         logger.info(f"Loading Diffusers pipeline from {model_id}")
    #         try:
    #             self.pipeline = LTXImageToVideoPipeline.from_pretrained(
    #                 model_id,
    #                 torch_dtype=dtype,
    #                 device_map=device_map,
    #             )
    #         except Exception:
    #             self.pipeline = LTXPipeline.from_pretrained(
    #                 model_id,
    #                 torch_dtype=dtype,
    #                 device_map=device_map,
    #             )
    #         if device_map is None:
    #             self.pipeline.to(self.device)
    #         logger.info("Diffusers LTX pipeline loaded.")
    #     except Exception as e:
    #         logger.error(f"Failed to load any LTX pipeline: {e}")
    #         raise RuntimeError(
    #             "无法加载 LTX 模型。请确认已安装官方 LTX-2 或 Diffusers，并正确配置 checkpoint 路径。"
    #         ) from e

    # ------------------------------------------------------------------
    # 单个 Chunk 生成
    # ------------------------------------------------------------------
    def _generate_chunk(
            self,
            prompt: str,
            condition_image=None,  # PIL.Image 或 None
            num_frames: int | None = None,
    ):
        from pathlib import Path
        import tempfile
        import torch
        from ltx_pipelines.utils.args import ImageConditioningInput  # 确认实际字段名

        num_frames = num_frames or int(self.cfg.generation.chunk_frames)
        width = int(self.cfg.generation.width)
        height = int(self.cfg.generation.height)
        fps = float(self.cfg.generation.fps)
        seed = int(self.cfg.generation.seed) + self._chunk_count

        # 官方: images: list[ImageConditioningInput]
        images = []
        if condition_image is not None:
            # ImageConditioningInput 的构造以你包内定义为准，常见是路径+起始帧
            # 先查: python -c "from ltx_pipelines.utils.args import ImageConditioningInput; import typing; print(ImageConditioningInput)"
            tmp = Path(tempfile.gettempdir()) / f"ltx_cond_{self._chunk_count}.png"
            condition_image.save(tmp)
            try:
                # 尝试常见形态（按你 inspect 结果改成正确的一种）
                images = [ImageConditioningInput(path=str(tmp), frame_idx=0)]
            except TypeError:
                images = [ImageConditioningInput(str(tmp), 0)]

        self.lora_mgr.apply_if_needed()

        with torch.inference_mode():
            video_iter, audio = self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                images=images,
                enhance_prompt=False,
            )
            # 必须在同一上下文里把 iterator 消费完，再转成 CPU/PIL
            frames = self._tensors_to_pil_frames(video_iter)

        return frames

    def _tensors_to_pil_frames(self, video_iter) -> list:
        """把官方 decode 的 iterator 收成 PIL 列表。"""
        import numpy as np
        from PIL import Image
        import torch

        frames = []
        for chunk in video_iter:
            # chunk 形状因版本而异，常见 [T,C,H,W] 或 [B,T,C,H,W] 或 [C,H,W]
            t = chunk.detach().float().cpu()
            if t.ndim == 5:
                t = t[0]
            if t.ndim == 3:
                t = t.unsqueeze(0)
            # 期望 T,C,H,W
            if t.shape[1] in (1, 3) or t.shape[1] > 4:
                pass
            elif t.shape[-1] in (1, 3):
                t = t.permute(0, 3, 1, 2)
            for i in range(t.shape[0]):
                f = t[i]
                if f.shape[0] in (1, 3):
                    f = f.permute(1, 2, 0)
                f = f.numpy()
                if f.max() <= 1.5:
                    f = (f * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    f = f.clip(0, 255).astype(np.uint8)
                if f.shape[-1] == 1:
                    f = np.repeat(f, 3, axis=-1)
                frames.append(Image.fromarray(f[..., :3]))
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
