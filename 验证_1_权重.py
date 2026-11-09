# -*- coding: utf-8 -*-
"""
验证 1：默认权重到底是谁 + 防呆校验是否真的生效。
运行：D:\\Anaconda3\\envs\\bearing\\python.exe 验证_1_权重.py

看三件事：
  1. 系统默认加载的权重文件是哪个、多大、哈希是什么（哈希相同=同一个文件）
  2. 这个权重的类别名是不是那 10 类（不是 COCO 的 80 类）
  3. 拿 COCO 权重去启动，会不会被拦住（应该报错，这就是防呆生效）
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'app'))          # 让脚本能 import app 里的模块

from config import CLASSES, resolve_weights    # noqa: E402
from detector import Detector                  # noqa: E402


def sha256_12(p):
    """算文件指纹的前 12 位。同一个权重文件，指纹一定相同。"""
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


print('=== 1. 默认权重 ===')
w = resolve_weights()
print('默认加载 :', w)
print('文件大小 : %.2f MB' % (w.stat().st_size / 1024 / 1024))
print('指纹     :', sha256_12(w))

print('\n=== 2. 模型规模与类别 ===')
det = Detector(w)
n_params = sum(p.numel() for p in det.model.model.parameters())
print('参数量   : {:,}（n 约 3.14M，s 约 11.14M，看数量级就知道是谁）'.format(n_params))
print('类别数   :', len(det.names))
print('类别清单 :', [det.names[i] for i in range(len(det.names))])
print('与 config 一致 :', sorted(det.names.values()) == sorted(CLASSES))

print('\n=== 3. n 备份还在不在 ===')
for p in (ROOT / 'weights' / 'best_v8n.pt', ROOT / 'runs' / 'detect' / 'apdd_n' / 'weights' / 'best.pt'):
    if p.exists():
        print('%s -> %.2f MB  指纹 %s' % (p.name, p.stat().st_size / 1024 / 1024, sha256_12(p)))
    else:
        print(p, '不存在')

print('\n=== 4. 拿 COCO 权重试启动（应该失败） ===')
coco = ROOT / 'yolov8n.pt'
if coco.exists():
    try:
        Detector(coco)
        print('没有被拦住 —— 防呆失效！')
    except RuntimeError as e:
        print('被拦住，报错内容：', str(e)[:80], '...')
else:
    print('yolov8n.pt 不在项目根目录，跳过这一项')

print('\n=== 5. 源码里还有没有 COCO 回退 ===')
cfg = (ROOT / 'app' / 'config.py').read_text(encoding='utf-8')
for kw in ('yolov8n.pt', 'yolo26n.pt'):
    print('config.py 出现 %-12s : %s（应为 False）' % (kw, cfg.count(kw)))
