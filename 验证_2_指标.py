# -*- coding: utf-8 -*-
"""
验证 2：论文/简历上的指标是不是真的。
运行：D:\\Anaconda3\\envs\\bearing\\python.exe 验证_2_指标.py（约 20 秒）

看三件事：
  1. 训练脚本自己写下的结果（metrics_summary.json）是多少
  2. results.csv 末轮（训练最后一轮）的 mAP50 / recall
  3. 我拿测试集重算，在部署阈值 conf=0.25、IoU=0.5 下的 P/R
     （论文里的 0.669/0.480 是"最佳 F1 点"，部署点是另一回事）
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'app'))

from detector import Detector  # noqa: E402

TEST_IMG = ROOT / 'datasets' / 'apdd' / 'images' / 'test'
TEST_LBL = ROOT / 'datasets' / 'apdd' / 'labels' / 'test'


def iou(a, b):
    """两个框的交并比 = 交面积 / 并面积。>=0.5 通常算"框对了"。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = w * h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt(lbl_path, w, h):
    """读 YOLO 标注：每行"类别 中心x 中心y 宽 高"（0~1 的比例），换算成像素框。"""
    out = []
    if not lbl_path.exists():
        return out
    for line in lbl_path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def pr_on_test(weights, iou_thr=0.5):
    """逐图比对：预测框与标注框同类别且 IoU>=阈值 记 TP，多出来的记 FP，没被命中的标注记 FN。"""
    d = Detector(weights)
    tp = fp = fn = 0
    for img_path in sorted(TEST_IMG.glob('*.jpg')):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = load_gt(TEST_LBL / (img_path.stem + '.txt'), w, h)
        r = d.infer(img)
        used = [False] * len(gt)
        for det in sorted(r.detections, key=lambda x: -x.conf):   # 先拿高置信度的框去匹配
            best_i, best_v = -1, 0.0
            for i, (c, box) in enumerate(gt):
                if used[i] or c != det.cls_id:
                    continue
                v = iou(det.xyxy, box)
                if v > best_v:
                    best_i, best_v = i, v
            if best_i >= 0 and best_v >= iou_thr:
                used[best_i] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for u in used if not u)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, p, rec


print('=== 1. 训练脚本记录的指标（metrics_summary.json） ===')
for tag in ('n', 's'):
    f = ROOT / 'runs' / 'detect' / ('apdd_' + tag) / 'metrics_summary.json'
    j = json.loads(f.read_text(encoding='utf-8'))
    for split in ('val', 'test'):
        m = j[split]
        print('%s %-4s: mAP50 %.3f  mAP50-95 %.3f  P %.3f  R %.3f' % (
            tag, split, m['mAP50'], m['mAP50_95'], m['precision'], m['recall']))
    print('%s 单张耗时: 均值 %.2f ms 中位 %.2f ms（ultralytics 内部计时）' % (
        tag, j['single_image_ms']['mean'], j['single_image_ms']['median']))

print('\n=== 2. results.csv 末轮 ===')
for tag in ('n', 's'):
    csv_p = ROOT / 'runs' / 'detect' / ('apdd_' + tag) / 'results.csv'
    lines = csv_p.read_text(encoding='utf-8').strip().splitlines()
    head = [c.strip() for c in lines[0].split(',')]
    last = [c.strip() for c in lines[-1].split(',')]
    idx = {k: i for i, k in enumerate(head)}
    print('%s: 轮数 %d  mAP50 %s  recall %s' % (
        tag, len(lines) - 1, last[idx['metrics/mAP50(B)']], last[idx['metrics/recall(B)']]))

print('\n=== 3. 部署阈值下重算 P/R（conf=0.25，IoU=0.5，共 %d 张测试图） ==='
      % len(list(TEST_IMG.glob('*.jpg'))))
for tag, wp in (('n', ROOT / 'weights' / 'best_v8n.pt'), ('s', ROOT / 'weights' / 'best.pt')):
    tp, fp, fn, p, rec = pr_on_test(wp)
    print('%s: TP %d  FP %d  FN %d  ->  P %.3f  R %.3f' % (tag, tp, fp, fn, p, rec))
