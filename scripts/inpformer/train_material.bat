@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat

for %%c in (material1 material2 material3 material4) do (
    echo ============================================================
    echo Training: %%c
    echo ============================================================
    python tools\inpformer\train.py ^
        --dataset mvtec ^
        --root ../datasets/material ^
        --category %%c ^
        --image-size 448 ^
        --train-batch-size 16 ^
        --eval-batch-size 16 ^
        --num-workers 8 ^
        --encoder-name dinov2reg_vit_base_14 ^
        --inp-num 6 ^
        --decoder-depth 8 ^
        --bottleneck-dropout 0.0 ^
        --max-steps 5000 ^
        --lr 1e-3 ^
        --weight-decay 1e-4 ^
        --warmup-iters 100 ^
        --early-stop-patience 20 ^
        --seed 42 ^
        --output-dir ./output_inpformer/material/%%c ^
        --project-name INP-Former_Anomalib
    echo.
)

echo ============================================================
echo All categories completed.
echo ============================================================
pause
