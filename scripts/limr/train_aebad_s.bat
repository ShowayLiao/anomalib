@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\train.py ^
    --dataset aebad_s ^
    --root I:\exp\datasets\AeBAD\AeBAD_S ^
    --category AeBAD_S ^
    --image-size 256 ^
    --train-batch-size 16 ^
    --eval-batch-size 16 ^
    --backbone resnet34 ^
    --alpha 1.75 ^
    --epochs 200 ^
    --output-dir ./output_limr/aebad_s
pause
