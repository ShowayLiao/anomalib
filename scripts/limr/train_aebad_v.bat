@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\train.py ^
    --dataset aebad_v ^
    --root ./datasets/AeBAD/AeBAD_V ^
    --category AeBAD_V ^
    --image-size 256 ^
    --train-batch-size 2 ^
    --eval-batch-size 16 ^
    --backbone resnet34 ^
    --alpha 1.75 ^
    --epochs 200 ^
    --output-dir ./output_limr/aebad_v
pause
