#!/bin/bash
# 多卡运行示例（torchrun）
# 用法: ./scripts/run_multi.sh 2   # 使用 2 张卡

set -e
cd "$(dirname "$0")/.."

NPROC=${1:-2}
shift || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NPROC \
    examples/interactive_cli.py \
    --config configs/default.yaml \
    --prompt "A young woman passionately singing a beautiful song, expressive face, cinematic lighting" \
    "$@"
