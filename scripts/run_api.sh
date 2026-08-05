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

# $1 = 模型根目录，$2 = 端口（可选）
if [ -n "${1:-}" ]; then
  MODEL_ROOT="$1"
elif [ -n "${LTX_MODEL_ROOT:-}" ]; then
  MODEL_ROOT="$LTX_MODEL_ROOT"
else
  echo "错误: 请传入模型根目录"
  echo "用法: bash scripts/run_api.sh /path/to/model_root [port]"
  exit 1
fi

if [ ! -d "$MODEL_ROOT" ]; then
  echo "错误: 模型根目录不存在: $MODEL_ROOT"
  exit 1
fi

MODEL_ROOT="$(cd "$MODEL_ROOT" && pwd)"
export LTX_MODEL_ROOT="$MODEL_ROOT"

PORT=${2:-${PORT:-8000}}

echo "=========================================="
echo " LTX Streaming API"
echo " MODEL_ROOT = $LTX_MODEL_ROOT"
echo " Listen     = http://${HOST}:${PORT}"
echo "=========================================="

# 不要给 uvicorn 传 --model-root；靠环境变量 LTX_MODEL_ROOT
.venv/bin/python -m uvicorn api.server:app --host "$HOST" --port "$PORT"
