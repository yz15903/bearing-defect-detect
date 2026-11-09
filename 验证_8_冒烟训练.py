# -*- coding: utf-8 -*-
"""
验证 8：从零冒烟——能不能只花 1 分钟就证明"训练这条链路是通的"。
运行：D:\\Anaconda3\\envs\\bearing\\python.exe 验证_8_冒烟训练.py（约 1 分钟）

做法：取 120 张训练图临时建一个小数据集，固定随机种子跑 1 个 epoch、imgsz 320。
      目的不是练出好模型，而是证明 data.yaml / 路径 / 依赖 / 显卡 / 保存流程都正常。
产出：runs\\detect\\smoke\\results.csv（1 行，含 loss 与 mAP）
"""
import random
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT / 'datasets' / 'smoke'
IMGS = SMOKE / 'images' / 'train'
LBLS = SMOKE / 'labels' / 'train'
N = 120

if SMOKE.exists():
    shutil.rmtree(SMOKE)
IMGS.mkdir(parents=True)
LBLS.mkdir(parents=True)

src_imgs = sorted((ROOT / 'datasets' / 'apdd' / 'images' / 'train').glob('*.jpg'))
random.seed(0)
for p in random.sample(src_imgs, N):
    shutil.copy2(p, IMGS / p.name)
    lbl = ROOT / 'datasets' / 'apdd' / 'labels' / 'train' / (p.stem + '.txt')
    if lbl.exists():
        shutil.copy2(lbl, LBLS / lbl.name)

yaml_path = SMOKE / 'data.yaml'
yaml_path.write_text(yaml.safe_dump({
    'path': str(SMOKE).replace('\\', '/'),
    'train': 'images/train',
    'val': 'images/train',
    'nc': 10,
    'names': {i: n for i, n in enumerate(
        ['budaodian', 'tufen', 'loudi', 'qikeng', 'pengshang',
         'tucengkailie', 'zangdian', 'jupi', 'aoxian', 'cahua'])},
}, allow_unicode=True), encoding='utf-8')

print('临时数据集：%d 张 -> %s' % (N, SMOKE))
print('开始 1 个 epoch（imgsz 320, batch 8, seed 0, workers 0）……')
model = YOLO(str(ROOT / 'weights' / 'best.pt'))     # 用已有权重当起点，省下载时间
model.train(data=str(yaml_path), epochs=1, imgsz=320, batch=8, workers=0,
            seed=0, project=str(ROOT / 'runs' / 'detect'), name='smoke', exist_ok=True,
            verbose=False, plots=False)

csv = ROOT / 'runs' / 'detect' / 'smoke' / 'results.csv'
print('\n=== 冒烟结果 ===')
print('results.csv 存在：', csv.exists())
if csv.exists():
    lines = [l for l in csv.read_text(encoding='utf-8').splitlines() if l.strip()]
    print('行数（表头+1 轮）：', len(lines))
    print('末行：', lines[-1][:120], '...')
print('\n[判定] 只要能跑完 1 轮并写出 results.csv，就说明训练链路通；mAP 很低是正常的（只喂了 120 张、320 分辨率）。')
