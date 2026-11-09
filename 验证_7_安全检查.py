# -*- coding: utf-8 -*-
"""
验证 7：安全与整洁——项目里有没有不该发出去的东西。
运行：D:\\Anaconda3\\envs\\bearing\\python.exe 验证_7_安全检查.py（5 秒）

查：手机号、邮箱、密钥、绝对路径、超大文件、临时残留。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATTERNS = {
    '手机号': re.compile(r'1[3-9]\d{9}'),
    '邮箱': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    '密钥/token': re.compile(r'(sk-[A-Za-z0-9]{16,}|api[_-]?key\s*=|secret\s*=|token\s*=\s*["\'][A-Za-z0-9]{16,})', re.I),
    '绝对用户路径': re.compile(r'C:\\Users\\[^\\]+', re.I),
}
SCAN_EXT = {'.py', '.md', '.txt', '.bat', '.yaml', '.yml', '.json', '.csv'}
SKIP_DIR = {'datasets', 'runs', 'weights', 'review_pack', '__pycache__'}

hits = {k: [] for k in PATTERNS}
for p in ROOT.rglob('*'):
    if not p.is_file() or p.suffix.lower() not in SCAN_EXT:
        continue
    if any(part in SKIP_DIR for part in p.parts):
        continue
    if p.stat().st_size > 2 * 1024 * 1024:
        continue
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    for name, rgx in PATTERNS.items():
        for m in rgx.finditer(text):
            hits[name].append((str(p.relative_to(ROOT)), m.group(0)[:40]))

print('=== 敏感信息扫描（已跳过 datasets / runs / weights / review_pack） ===')
for name, lst in hits.items():
    print('\n%s：%d 处' % (name, len(lst)))
    for f, s in lst[:6]:
        print('   %s  ->  %s' % (f, s))
    if len(lst) > 6:
        print('   ...（其余 %d 处省略）' % (len(lst) - 6))

print('\n=== 大文件（> 50 MB，传给别人前要剔除） ===')
big = [(p, p.stat().st_size / 1048576) for p in ROOT.rglob('*') if p.is_file() and p.stat().st_size > 50 * 1048576]
for p, mb in sorted(big, key=lambda x: -x[1])[:8]:
    print('   %-50s %.0f MB' % (str(p.relative_to(ROOT)), mb))
print('   合计 %.1f GB' % (sum(mb for _, mb in big) / 1024))

print('\n=== 目录体积 ===')
for d in ('app', 'weights', 'datasets', 'runs', 'review_pack', 'tools'):
    p = ROOT / d
    if not p.exists():
        continue
    size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    print('   %-14s %8.1f MB' % (d, size / 1048576))

print('\n=== 根目录一次性脚本（保留可复现，但别当程序入口） ===')
one_off = sorted(p.name for p in ROOT.glob('_*.py'))
print('   %d 个：%s' % (len(one_off), ', '.join(one_off)))

print('''
[判定]
- 手机号/邮箱应为 0 处（简历不在项目里；若命中说明写进了代码注释）。
- 绝对路径必然有（data.yaml、启动脚本），换电脑要改，这是已知项不是漏洞。
- 大文件（数据集 1.7 GB、权重）不发，只发代码 + 日志 + 截图。
''')
