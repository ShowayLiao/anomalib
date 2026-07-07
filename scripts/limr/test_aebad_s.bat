@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\test.py ^
    --dataset aebad_s ^
    --root I:\exp\datasets\AeBAD\AeBAD_S ^
    --category AeBAD_S ^
    --image-size 256 ^
    --eval-batch-size 16 ^
    --backbone resnet34 ^
    --alpha 1.75 ^
    --block-dropout 0.0 ^
    --block-ffn-dropout 0.0 ^
    --block-attn-dropout 0.0 ^
    --frozen-stages 3 ^
    --seed 54 ^
    --original-checkpoint I:\exp\LiMR\best_student_model_175.pth ^
    --output-dir ./output_limr_test/aebad_s
pause
