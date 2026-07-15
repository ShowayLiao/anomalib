#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python tools/supersimplenet/test.py \
    --dataset realiad \
    --root i:\exp\datasets\Real-IAD \
    --category end_cap \
    --realiad-resolution 256 \
    --realiad-json realiad_jsons/realiad_jsons_fuiad_0.0/end_cap.json \
    --image-size 256 \
    --train-batch-size 32 \
    --eval-batch-size 32 \
    --perlin-threshold 0.2 \
    --backbone wide_resnet50_2.tv_in1k \
    --checkpoint ./output_supersimplenet/realiad_end_cap/SuperSimpleNet/RealIAD/end_cap/v1/weights/lightning/model.ckpt \
    --output-dir ./output_supersimplenet/realiad_end_cap/test
