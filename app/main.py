# -*- coding: utf-8 -*-
"""程序入口：python app/main.py [--model 权重] [--cam 0] [--folder 图片目录]

双击 run_app.bat 也可以启动（会自动切到项目目录并使用 bearing 环境）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from config import resolve_weights  # noqa: E402
from detector import Detector  # noqa: E402
from ui_main import MainWindow, ensure_chinese_font  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, help='权重路径，默认自动查找 weights/best.pt')
    p.add_argument('--cam', type=int, default=0)
    p.add_argument('--folder', default=None, help='图片文件夹，填了就用回放模式')
    return p.parse_args()


def main():
    a = parse_args()
    weights = resolve_weights(a.model)
    detector = Detector(weights)
    print('加载权重：%s' % weights)
    ms = detector.warmup()
    print('预热完成，首帧耗时 %.0f ms' % ms)
    app = QApplication(sys.argv)
    fam = ensure_chinese_font(app)
    if fam:
        app.setFont(QFont(fam, 10))
    win = MainWindow(detector)
    if a.folder:
        win.folder = a.folder
        win.cmb_source.setCurrentText('图片文件夹')
    elif a.cam:
        win.cmb_source.setCurrentText('摄像头 %d' % a.cam)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
