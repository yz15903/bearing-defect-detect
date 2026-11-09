# -*- coding: utf-8 -*-
"""
验证 5：Bad Case 分析——到底是漏检多还是误检多，集中在哪些类别。
运行：D:\\Anaconda3\\envs\\bearing\\python.exe 验证_5_BadCase.py（约 1 分钟）

产出：
  runs\\badcase\\badcase.csv      每张测试图的 TP/FP/FN 明细
  runs\\badcase\\worst_fn_*.jpg   漏检最多的 6 张（绿框=人工标注，红框=模型预测）
  runs\\badcase\\worst_fp_*.jpg   误检最多的 6 张
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'app'))
from config import CLASSES  # noqa: E402
from detector import Detector  # noqa: E402

TEST_IMG = ROOT / 'datasets' / 'apdd' / 'images' / 'test'
TEST_LBL = ROOT / 'datasets' / 'apdd' / 'labels' / 'test'
OUT = ROOT / 'runs' / 'badcase'
OUT.mkdir(parents=True, exist_ok=True)
IOU_THR = 0.5


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = w * h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt(p, w, h):
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


det = Detector(ROOT / 'weights' / 'best.pt')
rows = []
miss_by_cls = Counter()
fp_by_cls = Counter()
for img_path in sorted(TEST_IMG.glob('*.jpg')):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gt = load_gt(TEST_LBL / (img_path.stem + '.txt'), w, h)
    res = det.infer(img)
    used = [False] * len(gt)
    fp_list = []
    for d in sorted(res.detections, key=lambda x: -x.conf):
        best_i, best_v = -1, 0.0
        for i, (c, box) in enumerate(gt):
            if used[i] or c != d.cls_id:
                continue
            v = iou(d.xyxy, box)
            if v > best_v:
                best_i, best_v = i, v
        if best_i >= 0 and best_v >= IOU_THR:
            used[best_i] = True
        else:
            fp_list.append(d)
            fp_by_cls[CLASSES[d.cls_id]] += 1
    miss = [gt[i] for i, u in enumerate(used) if not u]
    for c, _ in miss:
        miss_by_cls[CLASSES[c]] += 1
    rows.append({
        'image': img_path.name, 'gt': len(gt), 'pred': len(res.detections),
        'tp': len(gt) - len(miss), 'fp': len(fp_list), 'fn': len(miss),
        'missed_classes': '|'.join(CLASSES[c] for c, _ in miss),
        'fp_classes': '|'.join(CLASSES[d.cls_id] for d in fp_list),
    })

csv_path = OUT / 'badcase.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)

tp = sum(r['tp'] for r in rows)
fp = sum(r['fp'] for r in rows)
fn = sum(r['fn'] for r in rows)
print('=== 汇总（%d 张测试图，conf=0.25，IoU=0.5，同类别匹配） ===' % len(rows))
print('TP %d  FP %d  FN %d  ->  P %.3f  R %.3f' % (tp, fp, fn, tp / (tp + fp), tp / (tp + fn)))
print('完全正确的图：%d 张（%.1f%%）' % (sum(1 for r in rows if r['fp'] == 0 and r['fn'] == 0),
                                  100 * sum(1 for r in rows if r['fp'] == 0 and r['fn'] == 0) / len(rows)))
print('\n漏检最多的类别：', miss_by_cls.most_common(5))
print('误检最多的类别：', fp_by_cls.most_common(5))

print('\n=== 漏检最严重 6 张 ===')
for r in sorted(rows, key=lambda x: -x['fn'])[:6]:
    print('  %-28s GT %2d  预测 %2d  漏 %2d  漏掉:%s' % (r['image'], r['gt'], r['pred'], r['fn'], r['missed_classes'] or '-'))

print('\n=== 误检最严重 6 张 ===')
for r in sorted(rows, key=lambda x: -x['fp'])[:6]:
    print('  %-28s GT %2d  预测 %2d  误 %2d  误判成:%s' % (r['image'], r['gt'], r['pred'], r['fp'], r['fp_classes'] or '-'))

# 存图：绿框人工标注、红框模型预测，方便一页页看
for kind, key in (('fn', 'fn'), ('fp', 'fp')):
    for k, r in enumerate(sorted(rows, key=lambda x: -x[key])[:6], 1):
        img_path = TEST_IMG / r['image']
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for c, box in load_gt(TEST_LBL / (img_path.stem + '.txt'), w, h):
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 200, 0), 2)
            cv2.putText(img, CLASSES[c], (int(box[0]), max(12, int(box[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        res = det.infer(cv2.imread(str(img_path)))
        for d in res.detections:
            cv2.rectangle(img, (int(d.xyxy[0]), int(d.xyxy[1])), (int(d.xyxy[2]), int(d.xyxy[3])),
                          (0, 0, 230), 2)
            cv2.putText(img, '%s %.2f' % (CLASSES[d.cls_id], d.conf),
                        (int(d.xyxy[0]), int(d.xyxy[3]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1)
        cv2.imwrite(str(OUT / ('worst_%s_%d_%s' % (kind, k, r['image']))), img)

print('\n已导出：%s（明细 CSV）' % csv_path)
print('已导出：%s\\worst_fn_*.jpg / worst_fp_*.jpg（绿=人工标注，红=模型预测）' % OUT)
