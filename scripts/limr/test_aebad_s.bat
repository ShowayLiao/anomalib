@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\test.py ^
    --dataset aebad_s ^
    --root I:\exp\datasets\AeBAD\AeBAD_S ^
    --category AeBAD_S ^
    --image-size 256 ^
    --train-batch-size 16 ^
    --eval-batch-size 16 ^
    --backbone resnet34 ^
    --alpha 1.75 ^
    --aebad-s-domain-shift same ^
    --checkpoint .\output_limr\aebad_s\LiMR\AeBAD_S\AeBAD_S\v2\weights\lightning\model.ckpt ^
    --output-dir ./output_limr/aebad_s/test_same
pause
