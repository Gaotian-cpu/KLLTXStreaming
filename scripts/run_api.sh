#!/bin/bash
# 启动 HTTP API 服务（HLS m3u8 流）

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 需要系统已安装 ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found. HLS .ts encoding will fail."
  echo "  Ubuntu: sudo apt install ffmpeg"
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "Starting LTX Streaming API on http://${HOST}:${PORT}"
python -m uvicorn api.server:app --host "$HOST" --port "$PORT"
