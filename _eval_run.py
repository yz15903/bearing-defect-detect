# -*- coding: utf-8 -*-
"""按模型名评估：python _eval_run.py apdd_s"""
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(r'C:\Users\DELL\Downloads\bearing_detect')


def main(name):
    run = ROOT / 'runs' / 'detect' / name
    W = run / 'weights' / 'best.pt'
    DATA = ROOT / 'datasets' / 'apdd' / 'data.yaml'
    m = YOLO(str(W))
    out = {}
    for split in ('val', 'test'):
        r = m.val(data=str(DATA), split=split, imgsz=640, batch=16, plots=False, verbose=False)
        out[split] = {
            'mAP50': float(r.box.map50),
            'mAP50_95': float(r.box.map),
            'precision': float(r.box.mp),
            'recall': float(r.box.mr),
            'speed': {k: round(float(v), 3) for k, v in r.speed.items()},
        }
        print(split.upper(), json.dumps(out[split], ensure_ascii=False))

    img = sorted((ROOT / 'datasets' / 'apdd' / 'images' / 'test').glob('*.jpg'))[0]
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
    (run / 'metrics_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', name)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'apdd_n')
