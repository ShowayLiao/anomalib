@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat

set CATEGORIES=audiojack bottle_cap button_battery end_cap eraser fire_hood mint mounts pcb phone_battery plastic_nut plastic_plug porcelain_doll regulator rolled_strip_base sim_card_set switch tape terminalblock toothbrush toy_brick toy transistor1 u_block usb_adaptor usb vcpill wooden_beads woodstick zipper

for %%c in (%CATEGORIES%) do (
    echo ========================================
    echo Training category: %%c
    echo ========================================
    python tools\supersimplenet\train.py ^
        --dataset realiad ^
        --root i:\exp\datasets\Real-IAD ^
        --category %%c ^
        --realiad-resolution 256 ^
        --realiad-json realiad_jsons/realiad_jsons_fuiad_0.0/{category}.json ^
        --image-size 256 ^
        --train-batch-size 32 ^
        --eval-batch-size 32 ^
        --num-workers 8 ^
        --perlin-threshold 0.2 ^
        --backbone wide_resnet50_2.tv_in1k ^
        --layers layer2 layer3 ^
        --epochs 300 ^
        --early-stop-patience 10 ^
        --seed 42 ^
        --output-dir ./output_supersimplenet/realiad_%%c ^
        --project-name SuperSimpleNet
)
pause
