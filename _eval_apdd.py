# -*- coding: utf-8 -*-
"""评估 best.pt：验证集/测试集指标 + 单张推理耗时。"""
import json
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r'C:\Users\DELL\Downloads\bearing_detect')
W = ROOT / 'runs' / 'detect' / 'apdd_n' / 'weights' / 'best.pt'
DATA = ROOT / 'datasets' / 'apdd' / 'data.yaml'

if __name__ == '__main__':
    m = YOLO(str(W))
    out = {}
    for split in ('val', 'test'):
        r = m.val(data=str(DATA), split=split, imgsz=640, batch=16, plots=False, verbose=False)
        out[split] = {
            'mAP50': float(r.box.map50),
            'mAP50_95': float(r.box.map),
            'precision': float(r.box.mp),
            'recall': float(r.box.mr),
            'per_class_mAP50': {k: round(float(v), 4) for k, v in r.box.ap50_class.items()} if hasattr(r.box, 'ap50_class') else {},
            'speed': {k: round(float(v), 3) for k, v in r.speed.items()},
        }
        print(split.upper(), json.dumps(out[split], ensure_ascii=False))

    # 单张图片端到端耗时（含预处理/后处理）
    import cv2
    img = next((ROOT / 'datasets' / 'apdd' / 'images' / 'test').glob('*.jpg'))
    frame = cv2.imread(str(img))
    for _ in range(5):
        m.predict(frame, imgsz=640, verbose=False, device=0)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        m.predict(frame, imgsz=640, verbose=False, device=0)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    out['single_image_ms'] = {'mean': round(sum(ts) / len(ts), 2), 'median': round(ts[len(ts)//2], 2)}
    print('SINGLE', out['single_image_ms'])

    (ROOT / 'runs' / 'detect' / 'apdd_n' / 'metrics_summary.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('METRICS DONE')
