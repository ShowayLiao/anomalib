@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\supersimplenet\test.py ^
    --dataset aebad_v ^
    --root ./datasets/AeBAD/AeBAD_V ^
    --category AeBAD_V ^
    --image-size 256 ^
    --train-batch-size 32 ^
    --eval-batch-size 32 ^
    --aebad-v-domain-shift video1 ^
    --perlin-threshold 0.2 ^
    --backbone wide_resnet50_2.tv_in1k ^
    --checkpoint ./output_supersimplenet/aebad_v/SuperSimpleNet/AeBAD_V/AeBAD_V/v1/weights/lightning/model.ckpt ^
    --output-dir ./output_supersimplenet/aebad_v/test_video1
pause
