#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python tools/supersimplenet/train.py \
    --dataset folder \
    --root ./my_dataset \
    --category my_data \
    --folder-normal-dir good \
    --folder-abnormal-dir bad \
    --folder-mask-dir masks \
    --image-size 256 \
    --train-batch-size 32 \
    --eval-batch-size 32 \
    --num-workers 8 \
    --perlin-threshold 0.2 \
    --backbone wide_resnet50_2.tv_in1k \
    --layers layer2 layer3 \
    --epochs 300 \
    --early-stop-patience 10 \
    --seed 42 \
    --output-dir ./output_supersimplenet/my_dataset \
    --project-name SuperSimpleNet
