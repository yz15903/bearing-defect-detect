# -*- coding: utf-8 -*-
"""推理封装：加载 YOLO 权重，对单帧返回检测框列表。"""
from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics import YOLO

from config import CLASSES, IMGSZ_DEFAULT, class_cn

_FONT_CANDIDATES = [
    r'C:\Windows\Fonts\msyhbd.ttc',
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
]
_PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
            (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0)]


def _load_font(size):
    from PIL import ImageFont
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_boxes(frame, dets):
    """自己画框，标签用中文，避免 ultralytics 默认的英文类别名。"""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    font = _load_font(max(16, int(frame.shape[1] / 55)))
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.xyxy]
        color = _PALETTE[d.cls_id % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = '%s %.2f' % (class_cn(d.name), d.conf)
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 8)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 8], fill=color)
        draw.text((x1 + 4, ty + 2 - tb[1]), text, fill=(255, 255, 255), font=font)
    return np.array(img)[:, :, ::-1].copy()


@dataclass
class Detection:
    cls_id: int
    name: str
    conf: float
    xyxy: tuple

    @property
    def w(self):
        return self.xyxy[2] - self.xyxy[0]

    @property
    def h(self):
        return self.xyxy[3] - self.xyxy[1]


@dataclass
class InferResult:
    detections: list = field(default_factory=list)
    annotated: np.ndarray = None
    elapsed_ms: float = 0.0      # 推理 + 绘制
    infer_ms: float = 0.0        # 只算 model.predict
    draw_ms: float = 0.0         # 只算中文标注绘制


class Detector:
    def __init__(self, weights, imgsz=IMGSZ_DEFAULT, device=None, conf=0.25, iou=0.45):
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.names = self.model.names
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device if device is not None else (0 if torch.cuda.is_available() else 'cpu')
        got = [str(v) for v in (self.names.values() if isinstance(self.names, dict) else self.names)]
        if sorted(got) != sorted(CLASSES):
            raise RuntimeError('权重类别与 config.CLASSES 不一致，拒绝启动（可能加载了 COCO 占位权重）：%s'
                               % (got,))

    def infer(self, frame_bgr: np.ndarray) -> InferResult:
        import time
        t0 = time.perf_counter()
        res = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        t_pred = time.perf_counter()
        dets = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                name = self.names.get(int(k), str(k)) if isinstance(self.names, dict) else str(k)
                dets.append(Detection(int(k), name, float(c), (float(x1), float(y1), float(x2), float(y2))))
        annotated = draw_boxes(frame_bgr, dets)
        t1 = time.perf_counter()
        return InferResult(dets, annotated, (t1 - t0) * 1000,
                           (t_pred - t0) * 1000, (t1 - t_pred) * 1000)

    def warmup(self, size=640):
        """首帧要建 CUDA 上下文，耗时可达数秒，启动时先空跑一次。"""
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        return self.infer(dummy).elapsed_ms
