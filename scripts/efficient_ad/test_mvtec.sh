#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
CKPT="./output_efficient_ad/mvtec_bottle/EfficientAD/MVTecAD/bottle/v1/weights/lightning/model.ckpt"

python tools/efficient_ad/test.py \
    --dataset mvtec \
    --root ./datasets/MVTec \
    --category bottle \
    --image-size 256 \
    --eval-batch-size 16 \
    --model-size s \
    --checkpoint "$CKPT" \
    --output-dir ./output_efficient_ad/test_bottle
