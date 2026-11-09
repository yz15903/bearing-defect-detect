# -*- coding: utf-8 -*-
"""
验证 4：耗时到底多少，以及"帧间空闲会让显卡降频"这件事。
运行：
    D:\\Anaconda3\\envs\\bearing\\python.exe 验证_4_耗时.py s      （测当前默认的 YOLOv8s）
    D:\\Anaconda3\\envs\\bearing\\python.exe 验证_4_耗时.py n      （测备份的 YOLOv8n）

为什么分两次跑：两个模型塞进同一个进程会互相影响显存和显卡状态，
数值会失真。一次只测一个，才是干净的数。

看三件事：
  1. 连续跑（不等待）时推理是多少毫秒 —— 这是"模型本身有多快"
  2. 每帧之间等 0.1 / 0.2 秒时推理变多少 —— 这是演示突然变卡的真正原因（显卡降频）
  3. 界面口径（推理 + 中文标注绘制）是多少 —— 截图里"耗时"那一栏就是它
"""
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'app'))

from detector import Detector  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else 's'
weights = {'s': ROOT / 'weights' / 'best.pt', 'n': ROOT / 'weights' / 'best_v8n.pt'}[tag]

img = cv2.imread(str(sorted((ROOT / 'datasets' / 'apdd' / 'images' / 'test').glob('*.jpg'))[0]))
print('测试图尺寸', img.shape, '权重', weights.name, '(%s)' % tag)

d = Detector(weights)
for _ in range(4):          # 预热：首帧要建 CUDA 上下文，几百毫秒到几秒都正常，别被它吓到
    d.infer(img)

back = [d.infer(img).infer_ms for _ in range(20)]
print('\n连续跑：推理中位 %.1f ms' % statistics.median(back))

full = [d.infer(img).elapsed_ms for _ in range(20)]
print('界面口径（推理+绘制）中位 %.1f ms（紧接着连续跑测，此时显卡是高时钟状态）'
      % statistics.median(full))

for gap in (0.1, 0.2):
    time.sleep(2)           # 先空转 2 秒，让显卡进入低功耗状态，模拟演示时的停顿
    vals = []
    for _ in range(10):
        vals.append(d.infer(img).infer_ms)
        time.sleep(gap)
    print('每帧等 %.1f 秒：推理中位 %.1f ms' % (gap, statistics.median(vals)))
