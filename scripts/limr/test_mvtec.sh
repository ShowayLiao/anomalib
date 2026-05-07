#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
CKPT="./output_limr/mvtec_bottle/lightning_logs/version_0/checkpoints/epoch=xxx.ckpt"

python tools/limr/test.py \
    --dataset mvtec \
    --root ./datasets/MVTec \
    --category bottle \
    --image-size 224 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --backbone resnet50 \
    --alpha 1.75 \
    --checkpoint "$CKPT" \
    --output-dir ./output_limr/test_bottle
