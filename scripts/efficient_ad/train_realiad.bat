@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\efficient_ad\train.py ^
    --dataset realiad ^
    --root i:\exp\datasets\Real-IAD ^
    --category end_cap ^
    --realiad-resolution 256 ^
    --realiad-json realiad_jsons/realiad_jsons_fuiad_0.0/end_cap.json ^
    --image-size 256 ^
    --eval-batch-size 8 ^
    --num-workers 8 ^
    --model-size s ^
    --imagenet-dir ./datasets/imagenette ^
    --epochs 200 ^
    --lr 1e-4 ^
    --weight-decay 1e-5 ^
    --early-stop-patience 10 ^
    --seed 42 ^
    --output-dir ./output_efficient_ad/realiad_end_cap ^
    --project-name EfficientAD
pause
