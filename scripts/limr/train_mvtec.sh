#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python tools/limr/train.py \
    --dataset mvtec \
    --root ./datasets/MVTec \
    --category bottle \
    --image-size 224 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --backbone resnet50 \
    --alpha 1.75 \
    --epochs 200 \
    --lr 0.001 \
    --output-dir ./output_limr/mvtec_bottle
