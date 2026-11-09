# -*- coding: utf-8 -*-
"""对照组：YOLOv8s 在同一份 APDDD 数据、同一套超参上训练。"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r'C:\Users\DELL\Downloads\bearing_detect')

if __name__ == '__main__':
    model = YOLO(str(ROOT / 'yolov8s.pt'))
    model.train(
        data=str(ROOT / 'datasets' / 'apdd' / 'data.yaml'),
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project=str(ROOT / 'runs' / 'detect'),
        name='apdd_s',
        exist_ok=True,
        patience=12,
        close_mosaic=10,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        plots=True,
        val=True,
        verbose=True,
    )
    print('TRAIN S DONE')
