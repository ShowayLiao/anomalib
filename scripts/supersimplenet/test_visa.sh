#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
CKPT="./output_supersimplenet/visa_capsules/SuperSimpleNet/Visa/capsules/v1/weights/lightning/model.ckpt"

python tools/supersimplenet/test.py \
    --dataset visa \
    --root ./datasets/visa \
    --category capsules \
    --image-size 256 \
    --train-batch-size 32 \
    --eval-batch-size 32 \
    --perlin-threshold 0.2 \
    --backbone wide_resnet50_2.tv_in1k \
    --checkpoint "$CKPT" \
    --output-dir ./output_supersimplenet/visa_capsules/test
