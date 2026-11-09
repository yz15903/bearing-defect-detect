# -*- coding: utf-8 -*-
"""把 APDDD 的 VOC 标注转成 YOLO 格式，按 8:1:1 划分，并统一缩放到长边 1280 加快训练。"""
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

RAW = Path(r'C:\Users\DELL\Downloads\bearing_detect\datasets\apdd_raw')
DST = Path(r'C:\Users\DELL\Downloads\bearing_detect\datasets\apdd')
ANN = RAW / 'APSPC-Annotations' / 'Annotations'

CLASSES = ['budaodian', 'tufen', 'loudi', 'qikeng', 'pengshang',
           'tucengkailie', 'zangdian', 'jupi', 'aoxian', 'cahua']
IDX = {c: i for i, c in enumerate(CLASSES)}

for split in ('train', 'val', 'test'):
    (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)

items = []
for xml in sorted(ANN.glob('*.xml')):
    root = ET.parse(xml).getroot()
    fn = (root.findtext('filename') or '').strip()
    if not fn:
        continue
    src = RAW / 'APSPC1' / fn
    if not src.exists():
        src = RAW / 'APSPC2' / fn
    if not src.exists():
        continue
    objs = []
    for o in root.findall('object'):
        name = (o.findtext('name') or '').strip()
        if name not in IDX:
            continue
        bb = o.find('bndbox')
        if bb is None:
            continue
        x1 = float(bb.findtext('xmin')); y1 = float(bb.findtext('ymin'))
        x2 = float(bb.findtext('xmax')); y2 = float(bb.findtext('ymax'))
        if x2 <= x1 or y2 <= y1:
            continue
        objs.append((IDX[name], x1, y1, x2, y2))
    if objs:
        items.append((src, objs))

print('usable images with defects:', len(items))

random.seed(20261106)
random.shuffle(items)
n = len(items)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
splits = {'train': items[:n_train],
          'val': items[n_train:n_train + n_val],
          'test': items[n_train + n_val:]}

MAXSIDE = 1280
kept = 0
for split, sub in splits.items():
    for src, objs in tqdm(sub, desc=split):
        im = Image.open(src)
        im = im.convert('RGB')
        ow, oh = im.size
        scale = MAXSIDE / max(ow, oh) if max(ow, oh) > MAXSIDE else 1.0
        if scale < 1.0:
            im = im.resize((int(ow * scale), int(oh * scale)), Image.LANCZOS)
        stem = src.stem
        im.save(str(DST / 'images' / split / (stem + '.jpg')), quality=92)

        lines = []
        for cid, x1, y1, x2, y2 in objs:
            bw = (x2 - x1) / ow
            bh = (y2 - y1) / oh
            cx = ((x1 + x2) / 2.0) / ow
            cy = ((y1 + y2) / 2.0) / oh
            lines.append('%d %.6f %.6f %.6f %.6f' % (cid, cx, cy, bw, bh))
            kept += 1
        (DST / 'labels' / split / (stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')

print('boxes written:', kept)

yaml_path = DST / 'data.yaml'
yaml_path.write_text(
    '# APDDD 铝型材表面缺陷（10 类，VOC 转 YOLO）\n'
    'path: %s\n'
    'train: images/train\n'
    'val: images/val\n'
    'test: images/test\n'
    'nc: %d\n'
    'names:\n' % (str(DST).replace('\\', '/'), len(CLASSES))
    + ''.join('  %d: %s\n' % (i, c) for i, c in enumerate(CLASSES)),
    encoding='utf-8')
print('data.yaml ->', yaml_path)
