#!/usr/bin/env python3
"""LiMR 测试脚本：支持多种数据集的加载与评估。"""

import argparse
import json
import time
from pathlib import Path

import torch

from anomalib.data import (
    AeBAD_S, AeBAD_V,
    BMAD, BTech, Folder, Kolektor, MPDD, MVTecAD, MVTecAD2, MVTecLOCO,
    RealIAD, VAD, Visa,
)
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator
from anomalib.models import LiMR


def _build_standard_dataset(args):
    return {
        "root": args.root,
        "category": args.category,
        "image_size": (args.image_size, args.image_size),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }


def _build_realiad(args):
    return {
        "root": args.root,
        "category": args.category,
        "resolution": args.realiad_resolution,
        "json_path": args.realiad_json,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "test_split_mode": args.realiad_test_split_mode,
    }


def _build_kolektor(args):
    return {
        "root": args.root,
        "image_size": (args.image_size, args.image_size),
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }


def _build_folder(args):
    return {
        "root": args.root,
        "normal_dir": args.folder_normal_dir,
        "normal_test_dir": args.folder_normal_test_dir,
        "abnormal_dir": args.folder_abnormal_dir,
        "mask_dir": args.folder_mask_dir,
        "image_size": (args.image_size, args.image_size),
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "name": args.category or "folder_dataset",
    }


def _build_aebad_s(args):
    return {
        "root": args.root,
        "category": args.category,
        "domain_shift": args.aebad_s_domain_shift,
        "image_size": (args.image_size, args.image_size),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }


def _build_aebad_v(args):
    return {
        "root": args.root,
        "category": args.category,
        "domain_shift": args.aebad_v_domain_shift,
        "image_size": (args.image_size, args.image_size),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }


DATASET_REGISTRY = {
    "mvtec":       (MVTecAD,    _build_standard_dataset),
    "mvtecad2":    (MVTecAD2,   _build_standard_dataset),
    "mvtec_loco":  (MVTecLOCO,  _build_standard_dataset),
    "btech":       (BTech,      _build_standard_dataset),
    "bmad":        (BMAD,       _build_standard_dataset),
    "mpdd":        (MPDD,       _build_standard_dataset),
    "vad":         (VAD,        _build_standard_dataset),
    "visa":        (Visa,       _build_standard_dataset),
    "realiad":     (RealIAD,    _build_realiad),
    "kolektor":    (Kolektor,   _build_kolektor),
    "folder":      (Folder,     _build_folder),
    "aebad_s":     (AeBAD_S,    _build_aebad_s),
    "aebad_v":     (AeBAD_V,    _build_aebad_v),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"LiMR 测试脚本。支持数据集: {', '.join(DATASET_REGISTRY)}",
    )

    # 数据集
    parser.add_argument("--dataset", type=str, default="mvtec",
                        choices=list(DATASET_REGISTRY))
    parser.add_argument("--root", type=str, default="./datasets/MVTec")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    # RealIAD
    parser.add_argument("--realiad-resolution", type=str, default="256")
    parser.add_argument("--realiad-json", type=str, default=None)
    parser.add_argument("--realiad-test-split-mode", type=str, default="from_dir",
                        choices=["none", "from_dir", "synthetic"])

    # Folder
    parser.add_argument("--folder-normal-dir", type=str, default="normal")
    parser.add_argument("--folder-normal-test-dir", type=str, default=None)
    parser.add_argument("--folder-abnormal-dir", type=str, default="abnormal")
    parser.add_argument("--folder-mask-dir", type=str, default=None)

    # AeBAD
    parser.add_argument("--aebad-s-domain-shift", type=str, default="same",
                        choices=["same", "view"])
    parser.add_argument("--aebad-v-domain-shift", type=str, default="video1",
                        choices=["video1", "video2", "video3"])

    # 模型
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重文件路径")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--test-mask-ratio", type=float, default=0.0)
    parser.add_argument("--fpn-output-dim", type=int, nargs="+", default=None)

    # 输出
    parser.add_argument("--output-dir", type=str, default="./output_limr_test")

    # Benchmark
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measure-iterations", type=int, default=100)

    return parser.parse_args()


def measure_inference_speed(model, datamodule, device="cuda", warmup=10, iterations=100) -> dict:
    print("\n" + "=" * 80)
    print("推理速度测量")
    print("=" * 80)

    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    model = model.to(device)
    model.eval()
    use_cuda = device == "cuda" and torch.cuda.is_available()

    print(f"预热 ({warmup} iter)...")
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= warmup:
                break
            images = batch["image"].to(device)
            _ = model(images)

    print(f"测量 ({iterations} iter)...")
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
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                e3 = torch.cuda.Event(enable_timing=True)
                e0.record()
                images = batch["image"].to(device)
                _ = model(images)
                e1.record()
                torch.cuda.synchronize()
                images = batch["image"].to(device)
                e2.record()
                _ = model(images)
                e3.record()
                torch.cuda.synchronize()
                iter_e2e = e0.elapsed_time(e1) / 1000.0
                iter_pure = e2.elapsed_time(e3) / 1000.0
            else:
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
                print(f"  iter {i+1}/{iterations}: e2e={iter_e2e*1000:.2f}ms pure={iter_pure*1000:.2f}ms batch={batch_size}")

    avg_e2e = total_time_e2e / total_images * 1000
    avg_pure = total_time_pure / total_images * 1000
    print(f"\n【总体延迟】{avg_e2e:.2f} ms/img  {total_images/total_time_e2e:.2f} FPS")
    print(f"【纯推理】  {avg_pure:.2f} ms/img  {total_images/total_time_pure:.2f} FPS")

    return {
        "device": device,
        "total_images": total_images,
        "end_to_end": {"avg_ms_per_image": round(avg_e2e, 2), "fps": round(total_images / total_time_e2e, 2)},
        "pure_inference": {"avg_ms_per_image": round(avg_pure, 2), "fps": round(total_images / total_time_pure, 2)},
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cls, builder = DATASET_REGISTRY[args.dataset]
    datamodule = dataset_cls(**builder(args))

    print("=" * 80)
    print(f"LiMR 测试 - 数据集: {args.dataset}")
    print("=" * 80)
    for key, val in sorted(vars(args).items()):
        print(f"  {key}: {val}")
    print()

    evaluator = Evaluator(
        test_metrics=[
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        ],
    )

    model = LiMR(
        backbone=args.backbone,
        alpha=args.alpha,
        mask_ratio=args.mask_ratio,
        test_mask_ratio=args.test_mask_ratio,
        fpn_output_dim=tuple(args.fpn_output_dim) if args.fpn_output_dim else None,
        evaluator=evaluator,
    )

    engine = Engine()

    engine.test(model=model, datamodule=datamodule, ckpt_path=args.checkpoint)

    speed = measure_inference_speed(
        model=model,
        datamodule=datamodule,
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup=args.warmup_iterations,
        iterations=args.measure_iterations,
    )

    with open(output_dir / "inference_speed.json", "w") as f:
        json.dump(speed, f, indent=2)

    print(f"结果已保存: {output_dir}")


if __name__ == "__main__":
    main()
