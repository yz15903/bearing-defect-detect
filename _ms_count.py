# -*- coding: utf-8 -*-
import requests

BASE = 'https://modelscope.cn/api/v1/datasets'


def page(ns, name, root, pn):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, ns, name),
                     params={'Revision': 'master', 'Root': root, 'PageNumber': pn, 'PageSize': 100},
                     timeout=60)
    j = r.json()
    return (j.get('Data') or {}).get('Files') or []


def count(ns, name, root, cap=20):
    total = 0
    for pn in range(1, cap + 1):
        files = page(ns, name, root, pn)
        if not files:
            break
        total += len(files)
        if pn == 1:
            first = files[0]['Path']
    return total, (first if total else '')


def sample(ns, name, path):
    url = '%s/%s/%s/repo?Revision=master&FilePath=%s' % (BASE, ns, name, requests.utils.quote(path))
    return requests.get(url, timeout=60).text


for root in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    n, first = count('lemonbeach', 'steel_defect_dataset', root)
    print('lemonbeach', root, '->', n, '|', first)

txt = sample('lemonbeach', 'steel_defect_dataset', 'labels/train/0001388270-Raw01-f_00001.txt')
print('label sample:', repr(txt[:300]))

for root in ['APSPC1', 'APSPC2', 'APSPC-Annotations/Annotations']:
    n, first = count('youziye111', 'APDDD', root)
    print('APDDD', root, '->', n, '|', first)
