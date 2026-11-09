# -*- coding: utf-8 -*-
"""环境自检：确认 GPU 可用，并用项目自己的权重跑通一次真实推理。
用法：python check_env.py
"""
import os
import sys
import time

import torch
import ultralytics
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, 'weights', 'best.pt')
TEST_IMG = os.path.join(ROOT, 'datasets', 'apdd', 'images', 'test')


def main():
    print("=" * 62)
    print("Python      :", sys.version.split()[0])
    print("Torch       :", torch.__version__)
    print("CUDA 可用   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU         :", p.name, "%.1f GB" % (p.total_memory / 1024 ** 3),
              "sm_%d%d" % (p.major, p.minor))
    print("Ultralytics :", ultralytics.__version__)
    print("=" * 62)

    if not os.path.exists(WEIGHTS):
        print("找不到权重：", WEIGHTS)
        return
    model = YOLO(WEIGHTS)
    dev = 0 if torch.cuda.is_available() else "cpu"
    imgs = []
    if os.path.isdir(TEST_IMG):
        imgs = [os.path.join(TEST_IMG, n) for n in sorted(os.listdir(TEST_IMG)) if n.endswith('.jpg')][:1]
    if not imgs:
        print("找不到测试图，先跑一次数据集构建：", TEST_IMG)
        return
    img = imgs[0]

    r = model.predict(img, imgsz=640, conf=0.25, device=dev, verbose=False)[0]
    print("推理正常，检测到目标数：", len(r.boxes))
    for b in r.boxes[:5]:
        print("   %-10s conf=%.3f" % (model.names[int(b.cls)], float(b.conf)))

    model.predict(img, imgsz=640, device=dev, verbose=False)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(img, imgsz=640, device=dev, verbose=False)
    print("平均单张推理耗时(device=%s)：%.1f ms" % (dev, (time.perf_counter() - t0) / n * 1000))

    out = os.path.abspath("runs/check")
    os.makedirs(out, exist_ok=True)
    model.predict(img, imgsz=640, device=dev, save=True, project=out,
                  name="test", exist_ok=True, verbose=False)
    print("结果图已保存：%s" % os.path.join(out, "test"))
    print("自检通过，可以启动界面了：启动检测系统.bat")


if __name__ == "__main__":
    main()
