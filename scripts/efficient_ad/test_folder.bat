@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
set CKPT=./output_efficient_ad/my_dataset/EfficientAD/Folder/my_data/v1/weights/lightning/model.ckpt

python tools\efficient_ad\test.py ^
    --dataset folder ^
    --root ./my_dataset ^
    --category my_data ^
    --folder-normal-dir good ^
    --folder-abnormal-dir bad ^
    --folder-mask-dir masks ^
    --image-size 256 ^
    --eval-batch-size 16 ^
    --model-size s ^
    --checkpoint %CKPT% ^
    --output-dir ./output_efficient_ad/my_dataset/test
pause
