@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat

for %%c in (material1 material2 material3 material4) do (
    echo ============================================================
    echo Training: %%c
    echo ============================================================
    python tools\efficient_ad\train.py ^
        --dataset mvtec ^
        --root ../datasets/material ^
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
        --output-dir ./output_efficient_ad/material/%%c ^
        --project-name EfficientAD
    echo.
)

echo ============================================================
echo All categories completed.
echo ============================================================
pause
