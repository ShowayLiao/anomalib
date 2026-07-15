@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\efficient_ad\test.py ^
    --dataset aebad_v ^
    --root ./datasets/AeBAD/AeBAD_V ^
    --category AeBAD_V ^
    --image-size 256 ^
    --eval-batch-size 16 ^
    --aebad-v-domain-shift video1 ^
    --model-size s ^
    --checkpoint ./output_efficient_ad/aebad_v/EfficientAD/AeBAD_V/AeBAD_V/v1/weights/lightning/model.ckpt ^
    --output-dir ./output_efficient_ad/aebad_v/test_video1
pause
