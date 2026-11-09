# -*- coding: utf-8 -*-
"""判定逻辑与端面面积估算。

判据（当前实现）：
    面积比 = 缺陷框面积之和 / 端面像素面积
    NG 条件：面积比 >= Ta  或  存在单框置信度 >= Tc（任一命中即 NG，安全优先）
说明：端面面积在公开集上 186 张走 62% 兜底（霍夫圆拟合成功率 3/189），
      Ta=1.0%、Tc=0.60 均为初始值，尚未在验证集上扫点标定。
本模块不依赖界面，可直接单测。
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import FACE_AREA_FALLBACK


@dataclass
class JudgeResult:
    verdict: str = 'OK'
    defects: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    n_defects: int = 0
    area_px: float = 0.0          # 缺陷框面积之和（像素）
    face_area_px: float = 0.0     # 端面面积（像素）
    area_ratio: float = 0.0       # 面积比，小数（0.01 表示 1%）
    triggers: list = field(default_factory=list)   # 'area' / 'conf'
    elapsed_ms: float = 0.0
    infer_ms: float = 0.0
    draw_ms: float = 0.0
    face_method: str = 'fallback'   # 'circle' / 'fallback'

    @property
    def is_ng(self):
        return self.verdict == 'NG'


def judge(detections, conf_thr=0.25, conf_hard=0.6, area_ratio_thr=0.01, face_area_px=0.0):
    """按面积比与高置信单框双阈值判 OK/NG。

    conf_thr 只用于丢弃低置信框；area_ratio_thr 是小数（1% 传 0.01）。
    face_area_px 为 0 时退化为只按 Tc 判定，避免除零。
    """
    keep = [d for d in detections if d.conf >= conf_thr]
    counts = {}
    for d in keep:
        counts[d.name] = counts.get(d.name, 0) + 1

    area = sum(max(0.0, d.w) * max(0.0, d.h) for d in keep)
    ratio = (area / face_area_px) if face_area_px > 0 else 0.0

    triggers = []
    if face_area_px > 0 and ratio >= area_ratio_thr:
        triggers.append('area')
    if any(d.conf >= conf_hard for d in keep):
        triggers.append('conf')

    return JudgeResult(
        verdict='NG' if triggers else 'OK',
        defects=keep,
        counts=counts,
        n_defects=len(keep),
        area_px=area,
        face_area_px=face_area_px,
        area_ratio=ratio,
        triggers=triggers,
    )


def estimate_face_area(frame, fallback_ratio=FACE_AREA_FALLBACK, max_side=360):
    """估算端面在图像中的像素面积：优先霍夫圆拟合外圆，失败按图像面积比例兜底。

    霍夫圆在全分辨率图上会把主循环拖死，这里先缩到 max_side 再检测，半径按比例还原。
    返回 (面积, 方式)，方式为 'circle' 或 'fallback'。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(gray, 5)
    hh, ww = gray.shape[:2]
    short = min(hh, ww)
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=short,
            param1=120, param2=50,
            minRadius=int(short * 0.15), maxRadius=int(short * 0.55),
        )
    except Exception:
        circles = None
    if circles is not None and len(circles[0]) > 0:
        c = np.round(circles[0][0]).astype(int)
        r = float(c[2]) / max(scale, 1e-6)
        if r > 0:
            return float(np.pi * r * r), 'circle'
    return float(w * h * fallback_ratio), 'fallback'
