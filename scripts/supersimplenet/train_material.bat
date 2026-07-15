@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat

for %%c in (material1 material2 material3 material4) do (
    echo ============================================================
    echo Training: %%c
    echo ============================================================
    python tools\supersimplenet\train.py ^
        --dataset mvtec ^
        --root ../datasets/material ^
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
        --output-dir ./output_supersimplenet/material/%%c ^
        --project-name SuperSimpleNet
    echo.
)

echo ============================================================
echo All categories completed.
echo ============================================================
pause
