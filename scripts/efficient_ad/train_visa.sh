#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python tools/efficient_ad/train.py \
    --dataset visa \
    --root ./datasets/visa \
    --category capsules \
    --image-size 256 \
    --eval-batch-size 16 \
    --num-workers 8 \
    --model-size s \
    --imagenet-dir ./datasets/imagenette \
    --epochs 200 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --early-stop-patience 10 \
    --seed 42 \
    --output-dir ./output_efficient_ad/visa_capsules \
    --project-name EfficientAD
