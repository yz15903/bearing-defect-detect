# -*- coding: utf-8 -*-
"""从魔搭下载 APDDD 铝型材表面缺陷集（图片 + VOC 标注）。"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = 'https://modelscope.cn/api/v1/datasets'
NS, NAME = 'youziye111', 'APDDD'
OUT = Path(r'C:\Users\DELL\Downloads\bearing_detect\datasets\apdd_raw')
OUT.mkdir(parents=True, exist_ok=True)


def page(root, pn):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, NS, NAME),
                     params={'Revision': 'master', 'Root': root, 'PageNumber': pn, 'PageSize': 100},
                     timeout=60)
    return (r.json().get('Data') or {}).get('Files') or []


def list_all(root):
    out = []
    pn = 1
    while True:
        files = page(root, pn)
        if not files:
            break
        out.extend([f['Path'] for f in files if f.get('Type') == 'blob'])
        pn += 1
        if pn > 40:
            break
    return out


def download(path):
    dst = OUT / path
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = '%s/%s/%s/repo?Revision=master&FilePath=%s' % (BASE, NS, NAME, requests.utils.quote(path))
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and r.content:
                tmp = str(dst) + '.part'
                with open(tmp, 'wb') as f:
                    f.write(r.content)
                os.replace(tmp, dst)
                return len(r.content)
            time.sleep(1.5)
        except Exception:
            time.sleep(1.5)
    return -1


roots = ['APSPC1', 'APSPC2', 'APSPC-Annotations/Annotations']
all_paths = []
for root in roots:
    ps = list_all(root)
    print(root, 'files', len(ps), flush=True)
    all_paths.extend(ps)

print('total files', len(all_paths), flush=True)
t0 = time.time()
done = 0
failed = 0
total_bytes = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(download, p): p for p in all_paths}
    for fut in as_completed(futs):
        n = fut.result()
        done += 1
        if n < 0:
            failed += 1
            print('FAIL', futs[fut], flush=True)
        else:
            total_bytes += n
        if done % 200 == 0:
            print('%d/%d failed=%d %.1f MB %.0fs' % (done, len(all_paths), failed,
                                                    total_bytes / 1e6, time.time() - t0), flush=True)
print('DONE files=%d failed=%d bytes=%.1fMB time=%.0fs' % (done, failed, total_bytes / 1e6, time.time() - t0))
