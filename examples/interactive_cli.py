#!/usr/bin/env python3
"""
交互式命令行入口。

一边生成视频，一边在终端输入新指令：
  prompt: 一个女人正在深情地唱一首歌
  prompt: 她慢慢停下，开始背诵唐诗静夜思
  lora: /path/to/style.safetensors
  unload_lora
  stop
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# 保证能 import 到 src
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from rich.console import Console
from rich.panel import Panel

from src.config import load_config, merge_cli_overrides
from src.engine import StreamingEngine
from src.utils import setup_logger

console = Console()
logger = setup_logger("cli")


def input_loop(engine: StreamingEngine) -> None:
    """后台线程：读取用户指令。"""
    console.print(
        Panel.fit(
            "[bold green]LTX Streaming Interactive[/]\n"
            "命令：\n"
            "  prompt: <文本>     → 设置下一块提示词\n"
            "  lora: <路径>       → 下一 chunk 加载 LoRA\n"
            "  unload_lora        → 卸载 LoRA\n"
            "  stop               → 停止生成\n",
            title="控制台",
        )
    )

    while not engine.prompt_mgr.stopped:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue

        if line.lower() in ("stop", "quit", "exit", "q"):
            console.print("[yellow]Stopping...[/]")
            engine.stop()
            break
        elif line.lower().startswith("prompt:"):
            text = line[7:].strip()
            if text:
                engine.update_prompt(text)
                console.print(f"[cyan]Prompt queued:[/] {text[:100]}")
        elif line.lower().startswith("lora:"):
            path = line[5:].strip()
            engine.load_lora(path)
            console.print(f"[magenta]LoRA load requested:[/] {path}")
        elif line.lower() == "unload_lora":
            engine.unload_lora()
            console.print("[magenta]LoRA unload requested[/]")
        else:
            # 默认当作新提示词
            engine.update_prompt(line)
            console.print(f"[cyan]Prompt queued:[/] {line[:100]}")


def main():
    parser = argparse.ArgumentParser(description="LTX Streaming Interactive CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--prompt", type=str, default="A young woman looking at the camera, soft smile, cinematic lighting")
    parser.add_argument("--image", type=str, default=None, help="可选初始图片路径（I2V）")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    if args.width:
        overrides.setdefault("generation", {})["width"] = args.width
    if args.height:
        overrides.setdefault("generation", {})["height"] = args.height
    if args.max_chunks:
        overrides.setdefault("runtime", {})["max_chunks"] = args.max_chunks
    cfg = merge_cli_overrides(cfg, overrides)

    console.print(f"[bold]Loading engine with config:[/] {args.config}")
    engine = StreamingEngine(cfg)

    # 可选初始图像
    init_image = None
    if args.image:
        init_image = Image.open(args.image).convert("RGB")
        console.print(f"Using initial image: {args.image}")

    # 启动输入线程
    t = threading.Thread(target=input_loop, args=(engine,), daemon=True)
    t.start()

    # 主线程跑生成
    final = engine.run(
        initial_prompt=args.prompt,
        initial_image=init_image,
        max_chunks=args.max_chunks,
    )
    console.print(f"[bold green]Done.[/] Final video: {final}")


if __name__ == "__main__":
    main()
