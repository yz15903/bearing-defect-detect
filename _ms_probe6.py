# -*- coding: utf-8 -*-
import collections
import requests

BASE = 'https://modelscope.cn/api/v1/datasets'
NS, NAME = 'zyt1111', 'gear_defect'


def tree(root=''):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, NS, NAME),
                     params={'Revision': 'master', 'Root': root}, timeout=60)
    return (r.json().get('Data') or {}).get('Files') or []


def fetch(path):
    url = '%s/%s/%s/repo?Revision=master&FilePath=%s' % (BASE, NS, NAME, requests.utils.quote(path))
    return requests.get(url, timeout=60)


top = tree()
print('top entries', len(top))
for f in top[:12]:
    print('   ', f.get('Type'), f.get('Path'), f.get('Size'))

r = fetch('README.md')
print('---- README ----')
print(r.text[:1500])

for f in top:
    if f.get('Type') == 'tree':
        sub = tree(f['Path'])
        print('DIR', f['Path'], 'entries', len(sub))
        for s in sub[:6]:
            print('     ', s.get('Type'), s.get('Path'), s.get('Size'))
