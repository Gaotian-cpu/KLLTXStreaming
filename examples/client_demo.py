#!/usr/bin/env python3
"""
外部调用示例：提交任务 → 轮询状态 → 拿到 m3u8 地址播放。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--prompt", required=True, help="生成提示词")
    parser.add_argument("--image", default=None, help="可选图片路径")
    parser.add_argument("--segment-duration", type=float, default=3.0)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    base = args.base.rstrip("/")

    # 1. 提交任务
    data = {
        "prompt": args.prompt,
        "segment_duration": str(args.segment_duration),
        "max_chunks": str(args.max_chunks),
    }
    files = None
    if args.image:
        files = {"image": open(args.image, "rb")}

    print(f"[1] POST {base}/api/v1/tasks")
    r = requests.post(f"{base}/api/v1/tasks", data=data, files=files, timeout=60)
    if files:
        files["image"].close()
    r.raise_for_status()
    resp = r.json()
    task_id = resp["task_id"]
    print(f"    task_id = {task_id}")
    print(f"    query   = {resp.get('query_url')}")

    # 2. 轮询
    print("[2] Polling status...")
    stream_url = None
    while True:
        r = requests.get(f"{base}/api/v1/tasks/{task_id}", timeout=30)
        r.raise_for_status()
        info = r.json()
        status = info["status"]
        print(
            f"    status={status}  segments={info.get('segment_count', 0)}  "
            f"progress={info.get('progress', 0)}"
        )
        if info.get("stream_url"):
            stream_url = info["stream_url"]
            print(f"    stream_url = {stream_url}")
            print(f"    player_url = {info.get('player_url')}")
        if status in ("completed", "failed", "cancelled"):
            if status == "failed":
                print(f"    error: {info.get('error')}")
                sys.exit(1)
            break
        time.sleep(args.poll_interval)

    print("[3] Ready.")
    print(f"    在浏览器打开: {info.get('player_url')}")
    print(f"    或用播放器打开 m3u8: {stream_url}")


if __name__ == "__main__":
    main()
