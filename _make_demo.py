# -*- coding: utf-8 -*-
"""用训练好的权重在真实测试图上跑，出 1 张检测拼图 + 1 张系统界面截图。"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(r'C:\Users\DELL\Downloads\bearing_detect')
sys.path.insert(0, str(ROOT / 'app'))

import cv2
import numpy as np

from config import resolve_weights, SHOT_DIR
from detector import Detector
from judge import estimate_face_area, judge

DEMO = ROOT / 'runs' / 'demo'
DEMO.mkdir(parents=True, exist_ok=True)


def pick_images(det, test_dir, need=4):
    picks = []
    for p in sorted(test_dir.glob('*.jpg')):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        res = det.infer(frame)
        jr = judge(res.detections, 0.25, 0.60, 0.01, 0.0)
        if jr.n_defects >= 2:
            picks.append((p, frame, res.annotated, len(jr.defects)))
        if len(picks) >= need:
            break
    return picks


def make_grid(picks):
    tiles = []
    for i, (p, frame, ann, n) in enumerate(picks):
        img = cv2.resize(ann, (780, 585), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (780, 34), (32, 32, 32), -1)
        cv2.putText(img, 'TEST #%d   defects=%d' % (i + 1, n), (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) < 4:
        tiles.append(np.zeros((585, 780, 3), dtype=np.uint8))
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    grid = np.vstack([top, bot])
    out = DEMO / 'detect_grid.jpg'
    cv2.imwrite(str(out), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def make_ui_shot(det, sample):
    from PySide6.QtWidgets import QApplication
    from ui_main import MainWindow, ensure_chinese_font

    app = QApplication([])
    fam = ensure_chinese_font(app)
    if fam:
        app.setFont(app.font().family() and app.font() or app.font())
        f = app.font()
        f.setFamily(fam)
        app.setFont(f)

    db = DEMO / 'demo.db'
    if db.exists():
        db.unlink()
    win = MainWindow(det, db_path=str(db))
    win.cmb_source.setCurrentText('图片文件夹')
    win.resize(1400, 873)
    win.show()
    app.processEvents()

    folder = ROOT / 'datasets' / 'apdd' / 'images' / 'test'
    seq = sorted(folder.glob('*.jpg'))[:8]
    for i, p in enumerate(seq):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        res = det.infer(frame)
        face, how = estimate_face_area(frame)
        jr = judge(res.detections, 0.25, 0.60, 0.01, face)
        jr.elapsed_ms = res.elapsed_ms
        jr.infer_ms, jr.draw_ms, jr.face_method = res.infer_ms, res.draw_ms, how
        win.on_frame(res.annotated, jr, None)
        app.processEvents()
        time.sleep(0.05)

    # 定格到一张检出最多的样本
    best = sample
    res = det.infer(best[1])
    face, how = estimate_face_area(best[1])
    jr = judge(res.detections, 0.25, 0.60, 0.01, face)
    jr.elapsed_ms = res.elapsed_ms
    jr.infer_ms, jr.draw_ms, jr.face_method = res.infer_ms, res.draw_ms, how
    win.on_frame(res.annotated, jr, None)
    win.log('已载入测试样本：%s' % best[0].name)
    for _ in range(4):
        app.processEvents()
        time.sleep(0.1)
    out = SHOT_DIR / 'ui_real.png'
    win.grab().save(str(out))
    win.close()
    return out, jr


if __name__ == '__main__':
    w = resolve_weights()
    print('weights:', w)
    det = Detector(w)
    print('names:', det.names)
    det.warmup()
    picks = pick_images(det, ROOT / 'datasets' / 'apdd' / 'images' / 'test')
    print('picked:', [p[0].name for p in picks])
    g = make_grid(picks)
    print('grid:', g)
    ui, jr = make_ui_shot(det, picks[0])
    print('ui:', ui, 'verdict', jr.verdict, 'n', jr.n_defects)
    print('DEMO DONE')
