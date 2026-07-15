#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORIES=(audiojack bottle_cap button_battery end_cap eraser fire_hood mint mounts pcb phone_battery plastic_nut plastic_plug porcelain_doll regulator rolled_strip_base sim_card_set switch tape terminalblock toothbrush toy_brick toy transistor1 u_block usb_adaptor usb vcpill wooden_beads woodstick zipper)

for c in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo "Training category: $c"
    echo "========================================"
    python tools/efficient_ad/train.py \
        --dataset realiad \
        --root i:\exp\datasets\Real-IAD \
        --category "$c" \
        --realiad-resolution 256 \
        --realiad-json "realiad_jsons/realiad_jsons_fuiad_0.0/{category}.json" \
        --image-size 256 \
        --eval-batch-size 8 \
        --num-workers 8 \
        --model-size s \
        --imagenet-dir ./datasets/imagenette \
        --epochs 200 \
        --lr 1e-4 \
        --weight-decay 1e-5 \
        --early-stop-patience 10 \
        --seed 42 \
        --output-dir "./output_efficient_ad/realiad_${c}" \
        --project-name EfficientAD
done
