#!/bin/bash
# 单卡运行（H100 / H200）

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python examples/interactive_cli.py \
    --config configs/default.yaml \
    --prompt "A young woman passionately singing a beautiful song, expressive face, cinematic lighting" \
    "$@"
