# -*- coding: utf-8 -*-
import requests

BASE = 'https://modelscope.cn/api/v1/datasets'


def tree(ns, name, root=''):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, ns, name),
                     params={'Revision': 'master', 'Root': root}, timeout=30)
    j = r.json()
    return (j.get('Data') or {}).get('Files') or []


ns, name = 'lemonbeach', 'steel_defect_dataset'
for d in ['images', 'labels']:
    sub = tree(ns, name, d)
    print('DIR', d, 'entries', len(sub))
    for s in sub[:6]:
        print('   ', s.get('Type'), s.get('Path'), s.get('Size'))
        if s.get('Type') == 'tree':
            sub2 = tree(ns, name, s['Path'])
            print('     sub files', len(sub2))
            for x in sub2[:3]:
                print('       ', x.get('Path'), x.get('Size'))

r = requests.get('%s/%s/%s/repo?Revision=master&FilePath=README.md' % (BASE, ns, name), timeout=30)
print('---- README ----')
print(r.text[:1500])
