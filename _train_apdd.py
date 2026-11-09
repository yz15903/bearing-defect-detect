# -*- coding: utf-8 -*-
"""在 APDDD 铝型材表面缺陷集上训练 YOLOv8n。"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r'C:\Users\DELL\Downloads\bearing_detect')

if __name__ == '__main__':
    model = YOLO(str(ROOT / 'yolov8n.pt'))
    model.train(
        data=str(ROOT / 'datasets' / 'apdd' / 'data.yaml'),
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project=str(ROOT / 'runs' / 'detect'),
        name='apdd_n',
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
    print('TRAIN DONE')
