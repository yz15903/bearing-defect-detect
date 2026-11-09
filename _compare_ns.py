# -*- coding: utf-8 -*-
"""对比 YOLOv8n / YOLOv8s：精度、速度、参数量、体积，出表 + 出图。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(r'C:\Users\DELL\Downloads\bearing_detect')
RUNS = {'n': ROOT / 'runs' / 'detect' / 'apdd_n', 's': ROOT / 'runs' / 'detect' / 'apdd_s'}
OUT = ROOT / 'runs' / 'compare'
OUT.mkdir(parents=True, exist_ok=True)


def setup_cn_font():
    for p in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def main():
    setup_cn_font()
    rows = []
    for tag, run in RUNS.items():
        m = json.loads((run / 'metrics_summary.json').read_text(encoding='utf-8'))
        best = (run / 'weights' / 'best.pt')
        rows.append({
            'tag': tag,
            'val': m['val'], 'test': m['test'],
            'single_ms': m['single_image_ms'],
            'size_mb': round(best.stat().st_size / 1024 / 1024, 1),
        })
        print(tag, 'test mAP50', round(m['test']['mAP50'], 3), 'ms', m['single_image_ms'])

    # 表：markdown + csv
    head = ['模型', '测试集mAP50', '测试集mAP50-95', '测试集P', '测试集R',
            '验证集mAP50', '单帧predict墙钟ms', '权重MB']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        lines.append('| YOLOv8%s | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f | %.1f |' % (
            r['tag'], r['test']['mAP50'], r['test']['mAP50_95'], r['test']['precision'],
            r['test']['recall'], r['val']['mAP50'], r['single_ms']['median'], r['size_mb']))
    (OUT / 'compare.md').write_text('\n'.join(lines), encoding='utf-8')

    # 图：左精度右速度
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    labels = ['mAP50', 'mAP50-95', '精确率', '召回率']
    keys = [('mAP50', 'mAP50'), ('mAP50_95', 'mAP50-95'), ('precision', 'precision'), ('recall', 'recall')]
    x = range(len(labels))
    width = 0.36
    for i, r in enumerate(rows):
        vals = [r['test'][k] for k, _ in keys]
        b = axes[0].bar([p + (i - 0.5) * width for p in x], vals, width, label='YOLOv8' + r['tag'])
        axes[0].bar_label(b, fmt='%.3f', fontsize=8, padding=2)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_title('测试集精度对比（189 张）', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.25)

    ms = [r['single_ms']['median'] for r in rows]
    b2 = axes[1].bar(['YOLOv8' + r['tag'] for r in rows], ms, 0.45,
                     color=['#1f6feb', '#d9363e'])
    axes[1].bar_label(b2, fmt='%.1f ms', fontsize=9, padding=2)
    axes[1].set_ylabel('单帧 predict 墙钟 (ms)', fontsize=10)
    axes[1].set_title('推理速度对比（RTX 3060 Laptop 6G）', fontsize=12)
    axes[1].grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / 'compare.png')
    print('saved', OUT / 'compare.png')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
