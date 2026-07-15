@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\supersimplenet\train.py ^
    --dataset visa ^
    --root ./datasets/visa ^
    --category capsules ^
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
    --output-dir ./output_supersimplenet/visa_capsules ^
    --project-name SuperSimpleNet
pause
