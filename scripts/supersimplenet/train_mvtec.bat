@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\supersimplenet\train.py ^
    --dataset mvtec ^
    --root ./datasets/MVTec ^
    --category bottle ^
    --image-size 256 ^
    --train-batch-size 32 ^
    --eval-batch-size 32 ^
    --num-workers 8 ^
    --perlin-threshold 0.2 ^
    --backbone wide_resnet50_2.tv_in1k ^
    --layers layer2 layer3 ^
    --epochs 300 ^
    --early-stop-patience 10 ^
    --seed 42 ^
    --output-dir ./output_supersimplenet/mvtec_bottle ^
    --project-name SuperSimpleNet
pause
