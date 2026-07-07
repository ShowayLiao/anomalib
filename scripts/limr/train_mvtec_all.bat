@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
set CATEGORIES=bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper

for %%c in (%CATEGORIES%) do (
    echo ========================================
    echo Training category: %%c
    echo ========================================
    python tools\limr\train.py ^
        --dataset mvtec ^
        --root ./datasets/MVTec ^
        --category %%c ^
        --image-size 224 ^
        --train-batch-size 16 ^
        --eval-batch-size 16 ^
        --num-workers 6 ^
        --backbone resnet50 ^
        --alpha 1.75 ^
        --mask-ratio 0.4 ^
        --test-mask-ratio 0.0 ^
        --fpn-output-dim 64 128 256 512 ^
        --block-dropout 0.0 ^
        --block-ffn-dropout 0.0 ^
        --block-attn-dropout 0.0 ^
        --frozen-stages 3 ^
        --epochs 200 ^
        --lr 0.001 ^
        --weight-decay 0.05 ^
        --warmup-epochs 15 ^
        --early-stop-patience 10 ^
        --seed 54 ^
        --output-dir ./output_limr/mvtec_%%c ^
        --project-name LiMR
)
pause
