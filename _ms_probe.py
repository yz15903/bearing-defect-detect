# -*- coding: utf-8 -*-
import json
import time

import requests

BASE = 'https://modelscope.cn/api/v1/datasets'


def search(q, n=20):
    r = requests.get(BASE, params={'PageSize': n, 'PageNumber': 1, 'Query': q}, timeout=30)
    data = r.json().get('Data', [])
    items = data if isinstance(data, list) else (data.get('Datasets') or data.get('datasets') or [])
    out = []
    for d in items:
        out.append((str(d.get('Namespace', '')) + '/' + str(d.get('Name', '')),
                    str(d.get('Description') or '')[:50].replace('\n', ' ')))
    return out


for q in ['钢材', '金属', 'steel', 'scratch', '划痕', '锈', '缺陷数据集', 'YOLO 缺陷']:
    print('==', q)
    for name, desc in search(q):
        print('   ', name, '|', desc)


def tree(ns, name, root=''):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, ns, name),
                     params={'Revision': 'master', 'Root': root}, timeout=30)
    j = r.json()
    return (j.get('Data') or {}).get('Files') or []

print('---- APDDD APSPC1 ----')
for f in tree('youziye111', 'APDDD', 'APSPC1')[:10]:
    print('   ', f.get('Type'), f.get('Path'), f.get('Size'))
print('---- APDDD APSPC-Annotations ----')
for f in tree('youziye111', 'APDDD', 'APSPC-Annotations')[:10]:
    print('   ', f.get('Type'), f.get('Path'), f.get('Size'))

# speed test on one file
files = tree('youziye111', 'APDDD', 'APSPC1')
img = [f for f in files if str(f.get('Path', '')).lower().endswith(('.jpg', '.png'))]
if img:
    p = img[0]['Path']
    url = '%s/youziye111/APDDD/repo?Revision=master&FilePath=%s' % (BASE, requests.utils.quote(p))
    t0 = time.time()
    r = requests.get(url, timeout=60, stream=True)
    n = 0
    for chunk in r.iter_content(65536):
        n += len(chunk)
    dt = time.time() - t0
    print('SPEED', r.status_code, p, 'bytes=%d' % n, 'time=%.2fs' % dt, '%.0f KB/s' % (n / 1024 / max(dt, 0.001)))
