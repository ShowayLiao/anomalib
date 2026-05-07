@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\test.py ^
    --dataset realiad ^
    --root i:\exp\datasets\Real-IAD ^
    --category end_cap ^
    --realiad-resolution 1024 ^
    --realiad-json realiad_jsons/realiad_jsons_fuiad_0.0/end_cap.json ^
    --image-size 1024 ^
    --train-batch-size 8 ^
    --eval-batch-size 8 ^
    --backbone resnet50 ^
    --alpha 1.75 ^
    --checkpoint ./output_limr/realiad_end_cap/LiMR/RealIAD/end_cap/v8/weights/lightning/model.ckpt ^
    --output-dir ./output_limr/realiad_end_cap/test
pause
