# -*- coding: utf-8 -*-
import collections
import requests

BASE = 'https://modelscope.cn/api/v1/datasets'
NS, NAME = 'youziye111', 'APDDD'


def tree(root=''):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, NS, NAME),
                     params={'Revision': 'master', 'Root': root}, timeout=60)
    j = r.json()
    return (j.get('Data') or {}).get('Files') or []


ann = tree('APSPC-Annotations/Annotations')
print('annotations entries', len(ann), ann[:2] if ann else '')
xmls = [f['Path'] for f in ann if str(f.get('Path', '')).lower().endswith('.xml')]
print('xml count(shown)', len(xmls))

names = collections.Counter()
n_box = 0
for p in xmls[:40]:
    url = '%s/%s/%s/repo?Revision=master&FilePath=%s' % (BASE, NS, NAME, requests.utils.quote(p))
    r = requests.get(url, timeout=60)
    txt = r.text
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith('<name>') and s.endswith('</name>'):
            names[s[6:-7].strip()] += 1
            n_box += 1
print('boxes in 40 files:', n_box)
print('class names:', dict(names))

for d in ['APSPC1', 'APSPC2']:
    files = tree(d)
    print(d, 'entries', len(files))
