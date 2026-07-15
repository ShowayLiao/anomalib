#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORIES=(audiojack bottle_cap button_battery end_cap eraser fire_hood mint mounts pcb phone_battery plastic_nut plastic_plug porcelain_doll regulator rolled_strip_base sim_card_set switch tape terminalblock toothbrush toy_brick toy transistor1 u_block usb_adaptor usb vcpill wooden_beads woodstick zipper)

for c in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo "Training category: $c"
    echo "========================================"
    python tools/limr/train.py \
        --dataset realiad \
        --root i:\exp\datasets\Real-IAD \
        --category "$c" \
        --realiad-resolution 256 \
        --realiad-json "realiad_jsons/realiad_jsons_fuiad_0.0/{category}.json" \
        --image-size 256 \
        --train-batch-size 8 \
        --eval-batch-size 8 \
        --num-workers 6 \
        --backbone resnet50 \
        --alpha 1.75 \
        --mask-ratio 0.4 \
        --test-mask-ratio 0.0 \
        --fpn-output-dim 64 128 256 512 \
        --block-dropout 0.0 \
        --block-ffn-dropout 0.0 \
        --block-attn-dropout 0.0 \
        --frozen-stages 3 \
        --epochs 200 \
        --lr 0.001 \
        --weight-decay 0.05 \
        --warmup-epochs 15 \
        --early-stop-patience 10 \
        --seed 54 \
        --output-dir "./output_limr/realiad_${c}" \
        --project-name LiMR
done
