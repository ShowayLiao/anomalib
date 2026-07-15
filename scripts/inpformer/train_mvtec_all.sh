#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate
CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

for c in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo "Training category: $c"
    echo "========================================"
    python tools/inpformer/train.py \
        --dataset mvtec \
        --root ./datasets/MVTec \
        --category "$c" \
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
        --seed 42 \
        --output-dir "./output_inpformer/mvtec_${c}" \
        --project-name INP-Former_Anomalib
done
