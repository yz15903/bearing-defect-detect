# -*- coding: utf-8 -*-
"""训练脚本。
用法示例（RTX 3060 6GB，方案定稿：YOLOv8n 与 YOLOv8s 对比）：
    python train.py --model yolov8n --epochs 100 --batch 16 --name exp_n
    python train.py --model yolov8s --epochs 100 --batch 8  --name exp_s
两个模型都用同一份 data.yaml、同一套增强，跑完直接对比，结果写进论文第 4 章。
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n", help="yolov8n / yolov8s / 或直接给 .pt 路径")
    p.add_argument("--data", default="datasets/bearing/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="6GB 显存：n 用 16，s 用 8；OOM 就减半")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="exp1")
    p.add_argument("--resume", action="store_true", help="断点续训")
    return p.parse_args()


def main():
    a = parse_args()
    weights = a.model if a.model.endswith(".pt") else a.model + ".pt"
    model = YOLO(weights)

    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        device=0,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        # 数据增强：端面件有光照差异和来料姿态差异，这几项不要关
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.3, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,  # 端面上下翻转无物理意义，关掉
        mosaic=1.0,
        close_mosaic=15,         # 最后 15 轮关 mosaic，收敛更稳（写进论文）
        mixup=0.0,
        # 输出
        project="runs",
        name=a.name,
        exist_ok=True,
        plots=True,
        seed=42,
        resume=a.resume,
    )
    print("训练完成，最佳权重：runs/%s/weights/best.pt" % a.name)


if __name__ == "__main__":
    main()
