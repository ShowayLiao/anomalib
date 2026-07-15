@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
set CATEGORIES=bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper

for %%c in (%CATEGORIES%) do (
    echo ========================================
    echo Training category: %%c
    echo ========================================
    python tools\supersimplenet\train.py ^
        --dataset mvtec ^
        --root I:\exp\datasets\mvtec ^
        --category %%c ^
        --image-size 256 ^
        --train-batch-size 32 ^
        --eval-batch-size 32 ^
        --num-workers 8 ^
        --perlin-threshold 0.2 ^
        --backbone wide_resnet50_2.tv_in1k ^
        --layers layer2 layer3 ^
        --epochs 300 ^
        --early-stop-patience 10 ^
        --seed 42 ^
        --output-dir ./output_supersimplenet/mvtec_%%c ^
        --project-name SuperSimpleNet
)
pause
