@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\inpformer\train.py ^
    --dataset realiad ^
    --root i:\exp\datasets\Real-IAD ^
    --category end_cap ^
    --realiad-resolution 1024 ^
    --realiad-json realiad_jsons/realiad_jsons_fuiad_0.0/end_cap.json ^
    --image-size 1024 ^
    --train-batch-size 8 ^
    --eval-batch-size 8 ^
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
    --output-dir ./output_inpformer/realiad_end_cap ^
    --project-name INP-Former_Anomalib
pause
