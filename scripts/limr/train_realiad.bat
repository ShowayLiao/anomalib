@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
python tools\limr\train.py ^
    --dataset realiad ^
    --root i:\exp\datasets\Real-IAD ^
    --category end_cap ^
    --realiad-resolution 1024 ^
    --realiad-json realiad_jsons/realiad_jsons_fuiad_0.0/end_cap.json ^
    --image-size 1024 ^
    --train-batch-size 8 ^
    --eval-batch-size 8 ^
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
    --output-dir ./output_limr/realiad_end_cap ^
    --project-name LiMR
pause
