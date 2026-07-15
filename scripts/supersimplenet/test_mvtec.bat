@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
set CKPT=./output_supersimplenet/mvtec_bottle/SuperSimpleNet/MVTecAD/bottle/v1/weights/lightning/model.ckpt

python tools\supersimplenet\test.py ^
    --dataset mvtec ^
    --root ./datasets/MVTec ^
    --category bottle ^
    --image-size 256 ^
    --train-batch-size 32 ^
    --eval-batch-size 32 ^
    --perlin-threshold 0.2 ^
    --backbone wide_resnet50_2.tv_in1k ^
    --checkpoint %CKPT% ^
    --output-dir ./output_supersimplenet/test_bottle
pause
