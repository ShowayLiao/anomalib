#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORIES=(audiojack bottle_cap button_battery end_cap eraser fire_hood mint mounts pcb phone_battery plastic_nut plastic_plug porcelain_doll regulator rolled_strip_base sim_card_set switch tape terminalblock toothbrush toy_brick toy transistor1 u_block usb_adaptor usb vcpill wooden_beads woodstick zipper)

for c in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo "Training category: $c"
    echo "========================================"
    python tools/inpformer/train.py \
        --dataset realiad \
        --root i:\exp\datasets\Real-IAD \
        --category "$c" \
        --realiad-resolution 448 \
        --realiad-json "realiad_jsons/realiad_jsons_fuiad_0.0/{category}.json" \
        --image-size 448 \
        --train-batch-size 16 \
        --eval-batch-size 16 \
        --num-workers 8 \
        --encoder-name dinov2reg_vit_base_14 \
        --inp-num 6 \
        --decoder-depth 8 \
        --bottleneck-dropout 0.0 \
        --max-steps 5000 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --warmup-iters 100 \
        --early-stop-patience 20 \
        --seed 1 \
        --output-dir "./output_inpformer/realiad_${c}" \
        --project-name INP-Former_Anomalib
done
