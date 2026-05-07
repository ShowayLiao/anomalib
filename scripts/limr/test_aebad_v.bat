@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\test.py ^
    --dataset aebad_v ^
    --root ./datasets/AeBAD/AeBAD_V ^
    --category AeBAD_V ^
    --image-size 256 ^
    --train-batch-size 16 ^
    --eval-batch-size 16 ^
    --backbone resnet34 ^
    --alpha 1.75 ^
    --aebad-v-domain-shift video1 ^
    --checkpoint ./output_limr/aebad_v/LiMR/AeBAD_V/v1/weights/lightning/model.ckpt ^
    --output-dir ./output_limr/aebad_v/test_video1
pause
