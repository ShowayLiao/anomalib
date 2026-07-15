@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\efficient_ad\train.py ^
    --dataset aebad_v ^
    --root ./datasets/AeBAD/AeBAD_V ^
    --category AeBAD_V ^
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
    --output-dir ./output_efficient_ad/aebad_v ^
    --project-name EfficientAD
pause
