# -*- coding: utf-8 -*-
import io
import time

import requests

BASE = 'https://modelscope.cn/api/v1/datasets'


def tree(ns, name, root=''):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, ns, name),
                     params={'Revision': 'master', 'Root': root, 'PageSize': 100}, timeout=60)
    return (r.json().get('Data') or {}).get('Files') or []


def fetch(ns, name, path):
    url = '%s/%s/%s/repo?Revision=master&FilePath=%s' % (BASE, ns, name, requests.utils.quote(path))
    return requests.get(url, timeout=120)


# steel image real size
p = 'images/train/0001388270-Raw01-f_00001.jpg'
t0 = time.time()
r = fetch('lemonbeach', 'steel_defect_dataset', p)
dt = time.time() - t0
print('steel sample bytes', len(r.content), 'time %.2fs' % dt, 'speed %.0f KB/s' % (len(r.content) / 1024 / max(dt, .001)))
try:
    from PIL import Image
    im = Image.open(io.BytesIO(r.content))
    print('steel image size', im.size, im.mode)
except Exception as e:
    print('pil fail', e)

# PCB dataset
for ns, name in [('ModelBulider', 'PCB_DATASET_JSON'), ('AIinspect', 'MVTec_Screws')]:
    files = tree(ns, name)
    print('==', ns + '/' + name, 'entries', len(files))
    for f in files[:8]:
        print('   ', f.get('Type'), f.get('Path'), f.get('Size'))
    for f in files:
        if f.get('Type') == 'tree':
            sub = tree(ns, name, f['Path'])
            print('  DIR', f['Path'], len(sub), [s['Path'].split('/')[-1] for s in sub[:4]])
