# -*- coding: utf-8 -*-
import collections
import requests

BASE = 'https://modelscope.cn/api/v1/datasets'


def search(q, n=20):
    r = requests.get(BASE, params={'PageSize': n, 'PageNumber': 1, 'Query': q}, timeout=30)
    data = r.json().get('Data', [])
    items = data if isinstance(data, list) else (data.get('Datasets') or data.get('datasets') or [])
    return [(str(d.get('Namespace', '')) + '/' + str(d.get('Name', '')),
             str(d.get('Description') or '')[:60].replace('\n', ' ')) for d in items]


def tree(ns, name, root=''):
    r = requests.get('%s/%s/%s/repo/tree' % (BASE, ns, name),
                     params={'Revision': 'master', 'Root': root}, timeout=60)
    return (r.json().get('Data') or {}).get('Files') or []


def fetch(ns, name, path):
    url = '%s/%s/%s/repo?Revision=master&FilePath=%s' % (BASE, ns, name, requests.utils.quote(path))
    return requests.get(url, timeout=60).text


for q in ['轴承', '工件', '工业缺陷', '铝板', '齿轮', '螺丝', '划伤']:
    print('==', q, search(q)[:8])

for ns, name in [('chuanshanjia666', 'Aluminum-sheet-defect-detection-VOC-dataset'),
                 ('mobiwire', 'Mobile_PCB_Defect')]:
    files = tree(ns, name)
    print('==', ns + '/' + name, 'entries', len(files))
    for f in files[:8]:
        print('   ', f.get('Type'), f.get('Path'), f.get('Size'))
    for f in files:
        if f.get('Type') == 'tree' and ('Ann' in f['Path'] or 'ann' in f['Path'] or 'label' in f['Path']):
            sub = tree(ns, name, f['Path'])
            print('  DIR', f['Path'], len(sub))
            xmls = [s['Path'] for s in sub if str(s.get('Path', '')).lower().endswith(('.xml', '.txt'))]
            names = collections.Counter()
            for p in xmls[:25]:
                txt = fetch(ns, name, p)
                if p.endswith('.xml'):
                    for line in txt.splitlines():
                        s = line.strip()
                        if s.startswith('<name>') and s.endswith('</name>'):
                            names[s[6:-7].strip()] += 1
                else:
                    for line in txt.strip().splitlines():
                        parts = line.split()
                        if parts:
                            names['cls' + parts[0]] += 1
            print('   classes:', dict(names))
