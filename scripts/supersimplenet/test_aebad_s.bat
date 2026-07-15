@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\supersimplenet\test.py ^
    --dataset aebad_s ^
    --root I:\exp\datasets\AeBAD\AeBAD_S ^
    --category AeBAD_S ^
    --image-size 256 ^
    --train-batch-size 32 ^
    --eval-batch-size 32 ^
    --aebad-s-domain-shift same ^
    --perlin-threshold 0.2 ^
    --backbone wide_resnet50_2.tv_in1k ^
    --checkpoint ./output_supersimplenet/aebad_s/SuperSimpleNet/AeBAD_S/AeBAD_S/v1/weights/lightning/model.ckpt ^
    --output-dir ./output_supersimplenet/aebad_s/test
pause
