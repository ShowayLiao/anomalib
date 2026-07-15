#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python tools/efficient_ad/test.py \
    --dataset realiad \
    --root i:\exp\datasets\Real-IAD \
    --category end_cap \
    --realiad-resolution 256 \
    --realiad-json realiad_jsons/realiad_jsons_fuiad_0.0/end_cap.json \
    --image-size 256 \
    --eval-batch-size 8 \
    --model-size s \
    --checkpoint ./output_efficient_ad/realiad_end_cap/EfficientAD/RealIAD/end_cap/v1/weights/lightning/model.ckpt \
    --output-dir ./output_efficient_ad/realiad_end_cap/test
