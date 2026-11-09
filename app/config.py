# -*- coding: utf-8 -*-
"""全局配置：路径、类别名、权重查找规则、判定默认参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # 项目根目录 bearing_detect
APP_DIR = ROOT / 'app'
DB_PATH = APP_DIR / 'inspect.db'
SHOT_DIR = ROOT / 'runs' / 'ui_capture'
SHOT_DIR.mkdir(parents=True, exist_ok=True)

# 类别序号与名字必须和 datasets/apdd/data.yaml 中一致（10 类铝型材/金属表面缺陷）
CLASSES = [
    'budaodian', 'tufen', 'loudi', 'qikeng', 'pengshang',
    'tucengkailie', 'zangdian', 'jupi', 'aoxian', 'cahua',
]
CLASS_CN = {
    'budaodian': '不打点',
    'tufen': '涂粉',
    'loudi': '漏底',
    'qikeng': '起坑',
    'pengshang': '碰伤',
    'tucengkailie': '涂层开裂',
    'zangdian': '脏点',
    'jupi': '橘皮',
    'aoxian': '凹线',
    'cahua': '擦花',
}

CONF_DEFAULT = 0.25        # 低于此置信度的框直接丢弃
CONF_HARD_DEFAULT = 0.60   # 单框高于此置信度直接判 NG（Tc）
AREA_RATIO_DEFAULT = 1.0   # 缺陷面积占端面面积的比例阈值，单位 %（Ta）
IMGSZ_DEFAULT = 640
FACE_AREA_FALLBACK = 0.62  # 端面面积估不出来时，按图像面积的这个比例兜底


def class_cn(name):
    return CLASS_CN.get(name, name)


def resolve_weights(explicit=None):
    """按优先级找权重：显式指定 > weights/best.pt > 最新训练结果。

    不再回退到 COCO 预训练权重（80 类通用模型）：那会把鼠标键盘当缺陷判 NG。
    """
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands.append(ROOT / 'weights' / 'best.pt')
    runs = ROOT / 'runs'
    if runs.is_dir():
        for p in sorted(runs.rglob('weights/best.pt'), key=lambda x: x.stat().st_mtime, reverse=True):
            cands.append(p)
    for c in cands:
        if c.is_file() and c.stat().st_size > 1_000_000:
            return c
    raise FileNotFoundError('找不到训练权重，请把 best.pt 放到 weights/ 下')
