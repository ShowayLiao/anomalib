@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\efficient_ad\train.py ^
    --dataset aebad_s ^
    --root I:\exp\datasets\AeBAD\AeBAD_S ^
    --category AeBAD_S ^
    --image-size 256 ^
    --eval-batch-size 16 ^
    --num-workers 8 ^
    --aebad-s-domain-shift same ^
    --model-size s ^
    --imagenet-dir ./datasets/imagenette ^
    --epochs 200 ^
    --lr 1e-4 ^
    --weight-decay 1e-5 ^
    --early-stop-patience 10 ^
    --seed 42 ^
    --output-dir ./output_efficient_ad/aebad_s ^
    --project-name EfficientAD
pause
