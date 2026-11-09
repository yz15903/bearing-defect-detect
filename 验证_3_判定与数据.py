# -*- coding: utf-8 -*-
"""
验证 3：判定逻辑与数据集事实。
运行：D:\\Anaconda3\\envs\\bearing\\python.exe 验证_3_判定与数据.py（约 40 秒）

看四件事：
  1. 端面面积到底怎么来的：189 张测试图里霍夫圆成功几张（应为 3 张）
  2. 面积比的实际量级是多少，Ta 从 1% 调到 10% 能改多少判定结果
  3. 测试集里有没有"正常件"（没有正常件就测不了过杀率）
  4. 训练集/验证集/测试集之间有没有一模一样的图（有就说明指标偏乐观）
"""
import hashlib
import sqlite3
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'app'))

from detector import Detector            # noqa: E402
from judge import estimate_face_area, judge  # noqa: E402

DATA = ROOT / 'datasets' / 'apdd'

print('=== 1. 每个划分有多少张图、多少张是"空标注"（即没有缺陷的正常件） ===')
for split in ('train', 'val', 'test'):
    imgs = list((DATA / 'images' / split).glob('*.jpg'))
    empty = 0
    for p in imgs:
        lbl = DATA / 'labels' / split / (p.stem + '.txt')
        if (not lbl.exists()) or lbl.stat().st_size == 0:
            empty += 1
    print('%-5s 图片 %3d 张，空标注 %d 张' % (split, len(imgs), empty))

print('\n=== 2. 端面面积：霍夫圆成功率 ===')
test_imgs = sorted((DATA / 'images' / 'test').glob('*.jpg'))
face = []
for p in test_imgs:
    img = cv2.imread(str(p))
    area, method = estimate_face_area(img)
    face.append((area, method, img.shape[0] * img.shape[1]))
ok = sum(1 for _, m, _ in face if m == 'circle')
print('共 %d 张，霍夫圆成功 %d 张，兜底 %d 张' % (len(face), ok, len(face) - ok))
for i, (a, m, tot) in enumerate(face):
    if m == 'circle':
        print('  第 %d 张：拟合成功，端面/整图 = %.3f' % (i, a / tot))

print('\n=== 3. 面积比量级 + Ta 敏感度（用默认 Tc=0.60） ===')
d = Detector(ROOT / 'weights' / 'best.pt')
ratios, triggers, ng1, ng10 = [], {'area': 0, 'conf': 0, 'both': 0, 'none': 0}, 0, 0
for p, (fa, method, _) in zip(test_imgs, face):
    r = d.infer(cv2.imread(str(p)))
    j1 = judge(r.detections, face_area_px=fa, area_ratio_thr=0.01, conf_hard=0.60)
    j10 = judge(r.detections, face_area_px=fa, area_ratio_thr=0.10, conf_hard=0.60)
    ratios.append(j1.area_ratio)
    ng1 += 1 if j1.verdict == 'NG' else 0
    ng10 += 1 if j10.verdict == 'NG' else 0
    key = 'both' if ('area' in j1.triggers and 'conf' in j1.triggers) else (
        'area' if 'area' in j1.triggers else ('conf' if 'conf' in j1.triggers else 'none'))
    triggers[key] += 1
srt = sorted(ratios)
print('面积比 中位 %.1f%%  p90 %.1f%%  最大 %.1f%%' % (
    statistics.median(srt) * 100, srt[int(len(srt) * 0.9)] * 100, max(srt) * 100))
print('Ta=1%% 时 NG %d 张；Ta=10%% 时 NG %d 张（共 %d 张）' % (ng1, ng10, len(srt)))
print('触发来源：只有面积 %d，面积+置信度 %d，只有置信度 %d，都不触发 %d' % (
    triggers['area'], triggers['both'], triggers['conf'], triggers['none']))

print('\n=== 4. 数据库 ===')
for db in (ROOT / 'app' / 'inspect.db', ROOT / 'app' / 'inspect_coco_backup.db'):
    if db.exists():
        con = sqlite3.connect(str(db))
        n = con.execute('SELECT COUNT(*) FROM records').fetchone()[0]
        ng = con.execute("SELECT COUNT(*) FROM records WHERE verdict='NG'").fetchone()[0]
        src = con.execute('SELECT DISTINCT source FROM records').fetchall()
        print('%s: %d 条，NG %d，来源样例 %s' % (db.name, n, ng, [s[0] for s in src][:3]))
        con.close()
    else:
        print(db.name, '不存在')

print('\n=== 5. 跨划分重复图（MD5 全等） ===')
md5 = {}
for split in ('train', 'val', 'test'):
    for p in (DATA / 'images' / split).glob('*.jpg'):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        md5.setdefault(h, []).append(split + '/' + p.name)
dup_tt, dup_tv, dup_vt = 0, 0, 0
for h, lst in md5.items():
    splits = set(x.split('/')[0] for x in lst)
    if len(splits) > 1:
        if splits == {'train', 'test'}:
            dup_tt += 1
        elif splits == {'train', 'val'}:
            dup_tv += 1
        elif splits == {'val', 'test'}:
            dup_vt += 1
print('test 与 train 重复 %d 张，val 与 train 重复 %d 张，test 与 val 重复 %d 张' % (
    dup_tt, dup_tv, dup_vt))
