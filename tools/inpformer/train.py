#!/usr/bin/env python3
"""INP-Former 训练脚本：使用 INP-Former 模型进行异常检测模型的训练与评估。

支持的数据集:
  - MVTecAD    : MVTec Anomaly Detection 数据集
  - VisA       : Visual Anomaly 数据集
  - RealIAD    : Real Industrial Anomaly Detection 数据集

用法示例:
  # MVTec-AD 单类别训练
  python tools/inpformer/train.py --dataset MVTecAD --root ./datasets/MVTec --category bottle

  # VisA 训练
  python tools/inpformer/train.py --dataset Visa --root ./datasets/VisA --category candle

  # RealIAD 训练（指定分辨率与 JSON 路径）
  python tools/inpformer/train.py --dataset RealIAD --root ./datasets/Real-IAD --category end_cap --resolution 1024
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from anomalib.callbacks import TimerCallback
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import INP_Former

try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

    class WandbLogger:
        def __init__(self, *args, **kwargs): pass

    class wandb:
        @staticmethod
        def finish(): pass
        @staticmethod
        def log(*args, **kwargs): pass


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="INP-Former 模型训练与评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # 数据集参数
    # ============================================================
    parser.add_argument("--dataset", type=str, default="MVTecAD",
                        choices=["MVTecAD", "Visa", "RealIAD"],
                        help="数据集名称 (VisA 使用 Visa)")
    parser.add_argument("--root", type=str, default="./datasets/MVTec",
                        help="数据集根目录")
    parser.add_argument("--category", type=str, default="bottle",
                        help="训练类别")
    parser.add_argument("--resolution", type=str, default=None,
                        help="RealIAD 图像分辨率 (256/512/1024)，仅 RealIAD 使用")
    parser.add_argument("--json-path", type=str, default=None,
                        help="RealIAD JSON 路径（可选，不指定时使用数据集默认路径）")
    parser.add_argument("--image-size", type=int, default=448,
                        help="缩放尺寸")
    parser.add_argument("--crop-size", type=int, default=392,
                        help="中心裁剪尺寸")
    parser.add_argument("--train-batch-size", type=int, default=16,
                        help="训练批次大小")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="评估批次大小")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="数据加载线程数")

    # ============================================================
    # INP-Former 模型参数
    # ============================================================
    parser.add_argument("--encoder-name", type=str, default="dinov2reg_vit_base_14",
                        help="预训练编码器名称")
    parser.add_argument("--inp-num", type=int, default=6,
                        help="内在正常原型 (INP) 数量")
    parser.add_argument("--decoder-depth", type=int, default=8,
                        help="解码器 Transformer 层数")
    parser.add_argument("--bottleneck-dropout", type=float, default=0.0,
                        help="瓶颈层 Dropout 概率")

    # ============================================================
    # 训练参数
    # ============================================================
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="最大训练步数")
    parser.add_argument("--epochs", type=int, default=None,
                        help="最大训练轮数（设置后将覆盖 max-steps）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--warmup-iters", type=int, default=100,
                        help="学习率预热迭代数")
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="早停耐心值")

    # ============================================================
    # 输出参数
    # ============================================================
    parser.add_argument("--output-dir", type=str, default="./output_inpformer",
                        help="输出目录")
    parser.add_argument("--project-name", type=str, default="INP-Former_Anomalib",
                        help="WandB 项目名称")
    parser.add_argument("--run-name", type=str, default=None,
                        help="WandB 运行名称（默认自动生成）")

    # ============================================================
    # Benchmark 参数
    # ============================================================
    parser.add_argument("--warmup-iterations", type=int, default=10,
                        help="推理测速预热迭代次数")
    parser.add_argument("--measure-iterations", type=int, default=100,
                        help="推理测速测量迭代次数")

    return parser.parse_args()


def build_datamodule(args: argparse.Namespace):
    """根据参数构建数据集模块。

    Args:
        args: 命令行参数。

    Returns:
        Anomalib 数据模块实例。

    Raises:
        ValueError: 不支持的数据集名称。
    """
    datamodule_kwargs = {
        "root": args.root,
        "category": args.category,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }

    if args.dataset == "MVTecAD":
        datamodule_kwargs["image_size"] = (args.image_size, args.image_size)
        return MVTecAD(**datamodule_kwargs)

    if args.dataset == "Visa":
        datamodule_kwargs["image_size"] = (args.image_size, args.image_size)
        return Visa(**datamodule_kwargs)

    if args.dataset == "RealIAD":
        if args.resolution is None:
            print("警告: RealIAD 数据集建议指定 --resolution 参数，将默认使用 1024")
            args.resolution = "1024"
        datamodule_kwargs["resolution"] = args.resolution
        if args.json_path is not None:
            datamodule_kwargs["json_path"] = args.json_path
        datamodule_kwargs["test_split_mode"] = "from_dir"
        # RealIAD 使用独立的导入以避免循环依赖
        from anomalib.data import RealIAD
        return RealIAD(**datamodule_kwargs)

    raise ValueError(f"不支持的数据集: {args.dataset}")


def build_model(args: argparse.Namespace, evaluator) -> INP_Former:
    """构建 INP-Former 模型。

    Args:
        args: 命令行参数。
        evaluator: 评估器实例。

    Returns:
        ˙INP_Former 模型实例。
    """
    return INP_Former(
        encoder_name=args.encoder_name,
        inp_num=args.inp_num,
        bottleneck_dropout=args.bottleneck_dropout,
        decoder_depth=args.decoder_depth,
        evaluator=evaluator,
    )


def measure_inference_speed(
    model: INP_Former,
    datamodule,
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """测量模型推理速度。

    提供两种延迟指标:
      - 总体延迟 (end-to-end) : 包含数据传输 + 模型推理
      - 纯推理时间 (pure)     : 仅模型推理，不含传输

    Args:
        model:  已训练的模型。
        datamodule: 数据模块。
        device: 设备类型 ("cuda" 或 "cpu")。
        warmup: 预热迭代次数。
        iterations: 测量迭代次数。

    Returns:
        包含速度测量结果的字典。
    """
    print("\n" + "=" * 80)
    print("推理速度测量")
    print("=" * 80)

    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    model = model.to(device)
    model.eval()

    use_cuda = device == "cuda" and torch.cuda.is_available()

    # ---- 预热 ----
    print(f"预热阶段 ({warmup} 次迭代)...")
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= warmup:
                break
            images = batch["image"].to(device)
            _ = model(images)

    # ---- 测量 ----
    print(f"测量阶段 ({iterations} 次迭代)...")
    total_time_e2e = 0.0
    total_time_pure = 0.0
    total_images = 0

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= iterations:
                break

            batch_size = batch["image"].shape[0]
            total_images += batch_size

            if use_cuda:
                # GPU 高精度计时（cuda.Event）
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                e3 = torch.cuda.Event(enable_timing=True)

                # 总体延迟：数据传输 + 推理
                e0.record()
                images = batch["image"].to(device)
                _ = model(images)
                e1.record()
                torch.cuda.synchronize()

                # 纯推理：重新传输后仅推理
                images = batch["image"].to(device)
                e2.record()
                _ = model(images)
                e3.record()
                torch.cuda.synchronize()

                iter_e2e = e0.elapsed_time(e1) / 1000.0
                iter_pure = e2.elapsed_time(e3) / 1000.0
            else:
                # CPU 计时
                t0 = time.perf_counter()
                images = batch["image"].to(device)
                _ = model(images)
                iter_e2e = time.perf_counter() - t0

                images = batch["image"].to(device)
                t2 = time.perf_counter()
                _ = model(images)
                iter_pure = time.perf_counter() - t2

            total_time_e2e += iter_e2e
            total_time_pure += iter_pure

            if (i + 1) % 10 == 0:
                print(f"  iter {i+1}/{iterations}: "
                      f"e2e={iter_e2e*1000:.2f}ms, "
                      f"pure={iter_pure*1000:.2f}ms, "
                      f"batch={batch_size}")

    # ---- 汇总 ----
    avg_e2e_per_img = total_time_e2e / total_images * 1000
    avg_pure_per_img = total_time_pure / total_images * 1000
    fps_e2e = total_images / total_time_e2e
    fps_pure = total_images / total_time_pure

    print("\n" + "=" * 80)
    print("推理速度结果")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"总图像: {total_images}")
    print()
    print("【总体延迟（含数据传-推理）】")
    print(f"  平均每张: {avg_e2e_per_img:.2f} ms")
    print(f"  吞吐量: {fps_e2e:.2f} FPS")
    print()
    print("【纯推理时间（仅模型计算）】")
    print(f"  平均每张: {avg_pure_per_img:.2f} ms")
    print(f"  吞吐量: {fps_pure:.2f} FPS")
    print("=" * 80)

    return {
        "device": device,
        "total_images": total_images,
        "end_to_end": {
            "avg_ms_per_image": round(avg_e2e_per_img, 2),
            "fps": round(fps_e2e, 2),
        },
        "pure_inference": {
            "avg_ms_per_image": round(avg_pure_per_img, 2),
            "fps": round(fps_pure, 2),
        },
    }


def main():
    """主流程：构建数据集 → 构建模型 → 训练 → 测试 → 测速。"""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 打印配置 ----
    print("=" * 80)
    print("INP-Former 训练配置")
    print("=" * 80)
    for key, val in vars(args).items():
        print(f"  {key}: {val}")
    print()

    # ---- 构建数据集 ----
    print("=" * 80)
    print(f"构建数据集: {args.dataset}")
    print("=" * 80)
    datamodule = build_datamodule(args)

    # ---- 构建评估器 ----
    evaluator = Evaluator(
        test_metrics=[
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        ],
    )

    # ---- 构建模型 ----
    print("=" * 80)
    print("构建 INP-Former 模型")
    print("=" * 80)
    model = build_model(args, evaluator)

    # ---- 日志 ----
    run_name = args.run_name
    if run_name is None:
        run_name = f"INP-Former_{args.encoder_name}_{args.dataset}_{args.category}"

    logger = None
    if WANDB_AVAILABLE:
        logger = WandbLogger(
            project=args.project_name,
            name=run_name,
            config=vars(args),
        )

    # ---- 回调 ----
    callbacks = [
        TimerCallback(),
        EarlyStopping(
            monitor="train_loss_epoch",
            patience=args.early_stop_patience,
            mode="min",
            min_delta=0.001,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ---- 训练引擎 ----
    trainer_kwargs = {
        "default_root_dir": output_dir,
        "logger": logger,
        "callbacks": callbacks,
    }
    if args.epochs is not None:
        trainer_kwargs["max_epochs"] = args.epochs
    else:
        trainer_kwargs["max_steps"] = args.max_steps

    engine = Engine(**trainer_kwargs)

    # ---- 训练 ----
    print("=" * 80)
    print("开始训练")
    print("=" * 80)
    engine.fit(model=model, datamodule=datamodule)

    # ---- 测试 ----
    print("=" * 80)
    print("开始测试评估")
    print("=" * 80)
    engine.test(model=model, datamodule=datamodule)

    # ---- 推理速度测量 ----
    print("=" * 80)
    print("开始推理速度测量")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speed_results = measure_inference_speed(
        model=model,
        datamodule=datamodule,
        device=device,
        warmup=args.warmup_iterations,
        iterations=args.measure_iterations,
    )

    # ---- 保存结果 ----
    speed_path = output_dir / "inference_speed.json"
    with open(speed_path, "w") as f:
        json.dump(speed_results, f, indent=2)
    print(f"推理速度结果已保存: {speed_path}")

    if WANDB_AVAILABLE:
        wandb.log({"inference_speed": speed_results})
        wandb.finish()

    print("=" * 80)
    print("训练与评估全部完成")
    print(f"输出目录: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
