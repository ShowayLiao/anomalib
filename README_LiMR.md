# LiMR — anomalib 框架使用指南

## 快速启动

### 训练

```bash
# 激活虚拟环境
.venv\Scripts\activate.bat

# MVTec（单类别）
python tools\limr\train.py --dataset mvtec --root ./datasets/MVTec --category bottle --image-size 224 --backbone resnet50 --alpha 1.75 --epochs 200

# AeBAD-S
python tools\limr\train.py --dataset aebad_s --root I:\exp\datasets\AeBAD\AeBAD_S --category AeBAD_S --image-size 256 --backbone resnet34 --alpha 1.75 --fpn-output-dim 64 128 256 512 --block-dropout 0.0 --block-ffn-dropout 0.0 --block-attn-dropout 0.0 --frozen-stages 3 --epochs 200 --seed 54

# AeBAD-V
python tools\limr\train.py --dataset aebad_v --root I:\exp\datasets\AeBAD\AeBAD_V --category AeBAD_V --image-size 256 --backbone resnet34 --alpha 1.75 --fpn-output-dim 64 128 256 512 --block-dropout 0.0 --block-ffn-dropout 0.0 --block-attn-dropout 0.0 --frozen-stages 3 --epochs 200 --seed 54

# RealIAD
python tools\limr\train.py --dataset realiad --root ./datasets/RealIAD --category <category> --realiad-resolution 256 --realiad-json ./datasets/RealIAD/RealIAD.json

# VISA
python tools\limr\train.py --dataset visa --root ./datasets/VISA --category candle --image-size 224 --backbone resnet50

# Folder（自定义数据集）
python tools\limr\train.py --dataset folder --root ./datasets/my_data --folder-normal-dir normal --folder-abnormal-dir abnormal --folder-mask-dir mask --category my_dataset
```

### 测试

```bash
# anomalib ckpt 测试
python tools\limr\test.py --dataset mvtec --root ./datasets/MVTec --category bottle --image-size 224 --backbone resnet50 --alpha 1.75 --checkpoint ./output_limr/xxx.ckpt

# 原始 LiMR 权重测试（.pth）
python tools\limr\test.py --dataset aebad_s --root I:\exp\datasets\AeBAD\AeBAD_S --category AeBAD_S --image-size 256 --backbone resnet34 --alpha 1.75 --block-dropout 0.0 --block-ffn-dropout 0.0 --block-attn-dropout 0.0 --frozen-stages 3 --seed 54 --original-checkpoint I:\exp\LiMR\best_student_model_175.pth
```

> **注意**：测试时必须保持与训练相同的 `backbone`、`alpha`、`frozen-stages`、`fpn-output-dim`、`block-*dropout` 等参数，否则权重加载会不完整。

### 使用 .bat 脚本

| 脚本 | 用途 |
|------|------|
| `scripts\limr\train_aebad_s.bat` | 训练 AeBAD-S |
| `scripts\limr\train_aebad_v.bat` | 训练 AeBAD-V |
| `scripts\limr\train_mvtec.bat` | 训练 MVTec（单类别） |
| `scripts\limr\train_mvtec_all.bat` | 训练 MVTec（全类别） |
| `scripts\limr\train_visa.bat` | 训练 VISA |
| `scripts\limr\train_realiad.bat` | 训练 RealIAD |
| `scripts\limr\train_folder.bat` | 训练自定义数据集 |
| `scripts\limr\test_aebad_s.bat` | 测试 AeBAD-S |
| `scripts\limr\test_aebad_v.bat` | 测试 AeBAD-V |
| `scripts\limr\test_mvtec.bat` | 测试 MVTec |
| `scripts\limr\test_visa.bat` | 测试 VISA |
| `scripts\limr\test_realiad.bat` | 测试 RealIAD |
| `scripts\limr\test_folder.bat` | 测试自定义数据集 |

直接双击 `.bat` 文件即可运行，运行前需确保 `.venv` 虚拟环境已创建。

---

## 支持的数据集

| 参数值 `--dataset` | 数据集 |
|--------------------|--------|
| `mvtec` | MVTec AD |
| `mvtecad2` | MVTec AD 2 |
| `mvtec_loco` | MVTec LOCO |
| `btech` | BeanTech |
| `bmad` | BMAD |
| `mpdd` | MPDD |
| `vad` | VAD |
| `visa` | VISA |
| `realiad` | RealIAD |
| `kolektor` | KolektorSDD2 |
| `aebad_s` | AeBAD-S（静态图像 + domain shift） |
| `aebad_v` | AeBAD-V（视频帧） |
| `folder` | 自定义文件夹数据集 |

---

## 关键参数说明

### 数据集参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名称 | `mvtec` |
| `--root` | 数据集根目录 | `./datasets/MVTec` |
| `--category` | 子类别 / 物体类别 | `bottle` |
| `--image-size` | 输入图像尺寸（正方形） | `256` |
| `--train-batch-size` | 训练 batch size | `16` |
| `--eval-batch-size` | 评估 batch size | `16` |
| `--num-workers` | DataLoader 进程数 | `6` |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--backbone` | Teacher 网络 | `resnet50` |
| `--alpha` | Student 宽度系数（1.0=100%, 1.75=175%） | `1.75` |
| `--mask-ratio` | 训练时 mask 比例 | `0.4` |
| `--test-mask-ratio` | 测试时 mask 比例 | `0.0` |
| `--fpn-output-dim` | FPN 各层输出通道数 | 自动检测 |
| `--block-dropout` | MobileViTBlockv2 dropout | `0.1` |
| `--block-ffn-dropout` | FFN dropout | `0.0` |
| `--block-attn-dropout` | Attention dropout | `0.0` |
| `--frozen-stages` | 冻结的 encoder 阶段数（1-3） | `3` |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | `200` |
| `--lr` | 学习率 | `0.001` |
| `--weight-decay` | 权重衰减 | `0.05` |
| `--warmup-epochs` | 学习率预热轮数 | `15` |
| `--early-stop-patience` | 早停耐心值 | `10` |
| `--seed` | 随机种子 | `54` |

### 测试 / 输出参数

| 参数 | 说明 |
|------|------|
| `--checkpoint` | anomalib Lightning ckpt 路径（`.ckpt`），测试时使用 |
| `--original-checkpoint` | 原始 LiMR 仓库权重路径（`.pth`），测试时使用 |
| `--output-dir` | 输出目录 |
| `--project-name` | wandb 项目名 |

---

## 输出结构

训练完成后，`--output-dir` 下会生成：

```
output_limr/aebad_s/
├── lightning_logs/
│   └── version_0/
│       ├── checkpoints/        # 模型检查点 (.ckpt)
│       └── hparams.yaml        # 超参数记录
├── <dataset>/<category>/       # 按数据集的日志
│   └── metric_logs/
└── inference_speed.json        # 推理速度测量结果
```

---

## 常见问题

### 显存不足

1. 减小 `--train-batch-size`（如 `8` 或 `6`）
2. 减小 `--num-workers`
3. 将 `--backbone` 换为 `resnet18`

### 测试时权重加载报错

确保测试参数与训练参数一致。如果用 `--original-checkpoint` 加载旧仓库权重，必须使用 `--backbone resnet34`（旧仓库默认 backbone）。

### AeBAD 多 domain-shift 测试

AeBAD 测试会自动遍历所有 domain_shift 子目录（如 `same`, `background`, `illumination`, `view`），并在最后输出跨 shift 的平均指标。
