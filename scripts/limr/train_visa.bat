@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\train.py ^
    --dataset visa ^
    --root ./datasets/visa ^
    --category capsules ^
    --image-size 224 ^
    --train-batch-size 16 ^
    --eval-batch-size 16 ^
    --num-workers 6 ^
    --backbone resnet50 ^
    --alpha 1.75 ^
    --mask-ratio 0.4 ^
    --test-mask-ratio 0.0 ^
    --fpn-output-dim 64 128 256 512 ^
    --block-dropout 0.0 ^
    --block-ffn-dropout 0.0 ^
    --block-attn-dropout 0.0 ^
    --frozen-stages 3 ^
    --epochs 200 ^
    --lr 0.001 ^
    --weight-decay 0.05 ^
    --warmup-epochs 15 ^
    --early-stop-patience 10 ^
    --seed 54 ^
    --output-dir ./output_limr/visa_capsules ^
    --project-name LiMR
pause
