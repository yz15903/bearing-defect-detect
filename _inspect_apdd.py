# -*- coding: utf-8 -*-
import collections
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = Path(r'C:\Users\DELL\Downloads\bearing_detect\datasets\apdd_raw')
ANN = RAW / 'APSPC-Annotations' / 'Annotations'

cnt = collections.Counter()
files = sorted(ANN.glob('*.xml'))
imgs_missing = 0
sizes = collections.Counter()

for f in files:
    try:
        root = ET.parse(f).getroot()
    except Exception as e:
        print('BAD XML', f.name, e)
        continue
    for obj in root.findall('object'):
        n = (obj.findtext('name') or '').strip()
        cnt[n] += 1
    fn = (root.findtext('filename') or '').strip()
    if not fn:
        continue
    p1 = RAW / 'APSPC1' / fn
    p2 = RAW / 'APSPC2' / fn
    if p1.exists():
        sizes['APSPC1'] += 1
    elif p2.exists():
        sizes['APSPC2'] += 1
    else:
        imgs_missing += 1

print('xml files', len(files))
print('boxes by class:', dict(cnt))
print('matched images:', dict(sizes), 'missing:', imgs_missing)
print('images on disk:', len(list((RAW / 'APSPC1').glob('*.jpg'))), len(list((RAW / 'APSPC2').glob('*.jpg'))))

from PIL import Image
s = list((RAW / 'APSPC1').glob('*.jpg'))[:5]
for p in s:
    im = Image.open(p)
    print(p.name, im.size, p.stat().st_size // 1024, 'KB')
