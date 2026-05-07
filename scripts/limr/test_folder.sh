#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
CKPT="./output_limr/my_dataset/LiMR/Folder/my_data/v1/weights/lightning/model.ckpt"

python tools/limr/test.py \
    --dataset folder \
    --root ./my_dataset \
    --category my_data \
    --folder-normal-dir good \
    --folder-abnormal-dir bad \
    --folder-mask-dir masks \
    --image-size 224 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --backbone resnet50 \
    --alpha 1.75 \
    --checkpoint "$CKPT" \
    --output-dir ./output_limr/my_dataset/test
