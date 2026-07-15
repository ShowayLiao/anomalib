@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\efficient_ad\test.py ^
    --dataset aebad_s ^
    --root I:\exp\datasets\AeBAD\AeBAD_S ^
    --category AeBAD_S ^
    --image-size 256 ^
    --eval-batch-size 16 ^
    --aebad-s-domain-shift same ^
    --model-size s ^
    --checkpoint ./output_efficient_ad/aebad_s/EfficientAD/AeBAD_S/AeBAD_S/v1/weights/lightning/model.ckpt ^
    --output-dir ./output_efficient_ad/aebad_s/test
pause
