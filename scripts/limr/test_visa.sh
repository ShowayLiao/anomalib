#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
CKPT="./output_limr/visa_capsules/LiMR/Visa/capsules/v1/weights/lightning/model.ckpt"

python tools/limr/test.py \
    --dataset visa \
    --root ./datasets/visa \
    --category capsules \
    --image-size 224 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --backbone resnet50 \
    --alpha 1.75 \
    --checkpoint "$CKPT" \
    --output-dir ./output_limr/visa_capsules/test
