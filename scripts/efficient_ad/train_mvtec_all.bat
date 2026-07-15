@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
set CATEGORIES=bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper

for %%c in (%CATEGORIES%) do (
    echo ========================================
    echo Training category: %%c
    echo ========================================
    python tools\efficient_ad\train.py ^
        --dataset mvtec ^
        --root I:\exp\datasets\mvtec ^
        --category %%c ^
        --image-size 256 ^
        --eval-batch-size 16 ^
        --num-workers 8 ^
        --model-size s ^
        --imagenet-dir ./datasets/imagenette ^
        --epochs 200 ^
        --lr 1e-4 ^
        --weight-decay 1e-5 ^
        --early-stop-patience 10 ^
        --seed 42 ^
        --output-dir ./output_efficient_ad/mvtec_%%c ^
        --project-name EfficientAD
)
pause
