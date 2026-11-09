# -*- coding: utf-8 -*-
"""图像来源：工业相机/普通摄像头，或本地文件夹轮播（用于无相机演示）。"""
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


class CameraSource:
    """index 传整数＝本机摄像头；传字符串＝网络视频流地址（手机当相机就用这个）。"""

    def __init__(self, index=0, width=1280, height=720):
        if isinstance(index, str):
            self.cap = cv2.VideoCapture(index)
            self.kind = 'url'
            if not self.cap.isOpened():
                raise RuntimeError('打不开该视频流地址：%s' % index)
            return
        # Windows 下用 DirectShow，比默认后端启动快
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.kind = 'camera'
        if not self.cap.isOpened():
            raise RuntimeError('摄像头打开失败，换一个序号或改用图片文件夹模式')

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class FolderSource:
    def __init__(self, folder):
        self.files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
        if not self.files:
            raise RuntimeError('文件夹里没有图片：%s' % folder)
        self.i = 0
        self.kind = 'folder'

    def read(self):
        p = self.files[self.i % len(self.files)]
        self.i += 1
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def release(self):
        pass


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
