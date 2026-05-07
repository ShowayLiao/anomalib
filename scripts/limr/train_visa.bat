@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\train.py ^
    --dataset visa ^
    --root ./datasets/visa ^
    --category capsules ^
    --image-size 224 ^
    --train-batch-size 16 ^
    --eval-batch-size 16 ^
    --backbone resnet50 ^
    --alpha 1.75 ^
    --epochs 200 ^
    --output-dir ./output_limr/visa_capsules
pause
