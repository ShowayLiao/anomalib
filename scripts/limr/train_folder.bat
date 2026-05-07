@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\train.py ^
    --dataset folder ^
    --root ./my_dataset ^
    --category my_data ^
    --folder-normal-dir good ^
    --folder-abnormal-dir bad ^
    --folder-mask-dir masks ^
    --image-size 224 ^
    --train-batch-size 16 ^
    --eval-batch-size 16 ^
    --backbone resnet50 ^
    --alpha 1.75 ^
    --epochs 200 ^
    --output-dir ./output_limr/my_dataset
pause
