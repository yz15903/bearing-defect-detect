# -*- coding: utf-8 -*-
import time

import requests

REPO = 'KeenForgeAI/NEU-DET-corrected'
API = 'https://hf-mirror.com/api/datasets/' + REPO + '/tree/main'
RES = 'https://hf-mirror.com/datasets/' + REPO + '/resolve/main/'

r = requests.get(API, params={'path': 'images/train'}, timeout=60)
files = [x for x in r.json() if x.get('type') == 'file']
print('train images', len(files), files[0]['path'] if files else '')

r2 = requests.get(API, params={'path': 'labels/train'}, timeout=60)
labs = [x for x in r2.json() if x.get('type') == 'file']
print('train labels', len(labs))

sample = [f['path'] for f in files[:10]]
t0 = time.time()
ok = 0
total = 0
for p in sample:
    rr = requests.get(RES + p, timeout=60)
    if rr.status_code == 200:
        ok += 1
        total += len(rr.content)
dt = time.time() - t0
print('downloaded %d/%d, %d bytes, %.2fs, %.1f files/s' % (ok, len(sample), total, dt, ok / dt))

# label sample content for class ids
if labs:
    rr = requests.get(RES + labs[0]['path'], timeout=60)
    print('label sample:', labs[0]['path'])
    print(rr.text[:200])

rr = requests.get(RES + 'classes.txt', timeout=60)
print('classes.txt:', rr.text)
rr = requests.get(RES + 'data.yaml', timeout=60)
print('data.yaml:', rr.text)
