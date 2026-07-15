@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
set CKPT=./output_supersimplenet/my_dataset/SuperSimpleNet/Folder/my_data/v1/weights/lightning/model.ckpt

python tools\supersimplenet\test.py ^
    --dataset folder ^
    --root ./my_dataset ^
    --category my_data ^
    --folder-normal-dir good ^
    --folder-abnormal-dir bad ^
    --folder-mask-dir masks ^
    --image-size 256 ^
    --train-batch-size 32 ^
    --eval-batch-size 32 ^
    --perlin-threshold 0.2 ^
    --backbone wide_resnet50_2.tv_in1k ^
    --checkpoint %CKPT% ^
    --output-dir ./output_supersimplenet/my_dataset/test
pause
