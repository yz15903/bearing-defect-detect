# -*- coding: utf-8 -*-
import requests

BASE = 'https://modelscope.cn/api/v1/datasets'


def tree(ns, name, root=''):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, ns, name),
                     params={'Revision': 'master', 'Root': root}, timeout=30)
    j = r.json()
    return (j.get('Data') or {}).get('Files') or [], r.status_code


for ns, name in [('xiaopan999', 'GC10-DET'), ('liangziyue', 'steel_ball'),
                 ('lemonbeach', 'steel_defect_dataset'), ('Voxel51', 'severstal_steel_defects')]:
    files, code = tree(ns, name)
    print('==', ns + '/' + name, code, 'entries', len(files))
    for f in files[:12]:
        print('   ', f.get('Type'), f.get('Path'), f.get('Size'))

print('---- drill GC10-DET dirs ----')
files, _ = tree('xiaopan999', 'GC10-DET')
for f in files:
    if f.get('Type') == 'tree':
        sub, _ = tree('xiaopan999', 'GC10-DET', f['Path'])
        print('  DIR', f['Path'], 'files', len(sub))
        for s in sub[:5]:
            print('     ', s.get('Type'), s.get('Path'), s.get('Size'))

print('---- steel_ball drill ----')
files, _ = tree('liangziyue', 'steel_ball')
for f in files:
    if f.get('Type') == 'tree':
        sub, _ = tree('liangziyue', 'steel_ball', f['Path'])
        print('  DIR', f['Path'], 'files', len(sub))
        for s in sub[:5]:
            print('     ', s.get('Type'), s.get('Path'), s.get('Size'))
