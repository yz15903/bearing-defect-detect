# -*- coding: utf-8 -*-
"""检测系统主界面：图像显示、判定结果、参数设置、记录查询。"""
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from camera import CameraSource, FolderSource
from config import (AREA_RATIO_DEFAULT, CONF_DEFAULT, CONF_HARD_DEFAULT, FACE_AREA_FALLBACK,
                    IMGSZ_DEFAULT, SHOT_DIR, class_cn)
from db import InspectionDB
from judge import estimate_face_area, judge

UI_FONT = 'Microsoft YaHei'
FONT_FILES = [
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\msyhbd.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
]


def ensure_chinese_font(app):
    """有些环境 Qt 枚举不到系统字体，需要按文件显式加载，否则中文显示成方框。"""
    fams = QFontDatabase.families()
    for key in ('YaHei', 'Hei', 'Song', 'Sim'):
        hit = [f for f in fams if key in f]
        if hit:
            return hit[0]
    for p in FONT_FILES:
        if os.path.isfile(p):
            fid = QFontDatabase.addApplicationFont(p)
            if fid != -1:
                loaded = QFontDatabase.applicationFontFamilies(fid)
                if loaded:
                    return loaded[0]
    return None


class InferenceWorker(QThread):
    """采集 + 推理线程，避免阻塞界面。"""
    frame_ready = Signal(object, object, float)   # 标注图, JudgeResult, fps
    failed = Signal(str)

    def __init__(self, source, detector, conf, conf_hard, area_thr, pace=0.0, parent=None):
        super().__init__(parent)
        self.source = source
        self.detector = detector
        self.conf = conf
        self.conf_hard = conf_hard
        self.area_thr = area_thr
        self.pace = pace
        self._face_area = 0.0
        self._face_method = 'fallback'
        self._running = True

    def run(self):
        t_prev = time.perf_counter()
        while self._running:
            try:
                frame = self.source.read()
            except Exception as e:
                self.failed.emit(str(e))
                return
            if frame is None:
                self.msleep(30)
                continue
            try:
                if self._face_area <= 0:
                    self._face_area, self._face_method = estimate_face_area(frame)
                res = self.detector.infer(frame)
                jr = judge(res.detections, self.conf, self.conf_hard, self.area_thr, self._face_area)
                jr.elapsed_ms = res.elapsed_ms
                jr.infer_ms = res.infer_ms
                jr.draw_ms = res.draw_ms
                jr.face_method = self._face_method
            except Exception as e:
                import traceback
                self.failed.emit('%s（%s）' % (e, traceback.format_exc(limit=1).strip().splitlines()[-1]))
                return
            now = time.perf_counter()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            if self._running:
                self.frame_ready.emit(res.annotated, jr, fps)
            if self.pace > 0:
                self.msleep(int(self.pace * 1000))

    def stop(self):
        self._running = False
        self.wait(3000)


class MainWindow(QMainWindow):
    def __init__(self, detector, db_path=None):
        super().__init__()
        self.detector = detector
        self.db = InspectionDB(db_path) if db_path else InspectionDB()
        self.worker = None
        self.source = None
        self.folder = None
        self.last_record = 0.0
        self.last_verdict = None
        self._build()

    # ---------- 界面 ----------
    def _build(self):
        self.setWindowTitle('金属表面缺陷检测系统（APDDD 公开集流程验证）')
        self.resize(1400, 860)
        font = QApplication.font()
        font.setPointSize(10)
        self.setFont(font)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # 顶部工具条
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel('图像源'))
        self.cmb_source = QComboBox()
        self.cmb_source.addItems(['摄像头 0', '摄像头 1', '手机网络摄像头', '图片文件夹'])
        self.cmb_source.setFixedWidth(130)
        self.cmb_source.currentTextChanged.connect(self.on_source_changed)
        bar.addWidget(self.cmb_source)

        self.ed_url = QLineEdit('http://192.168.1.100:8080/video')
        self.ed_url.setFixedWidth(250)
        self.ed_url.setToolTip('手机端 IP Webcam 显示的地址，例如 http://192.168.1.100:8080/video')
        self.ed_url.setVisible(False)
        bar.addWidget(self.ed_url)

        self.btn_folder = QPushButton('选择文件夹')
        self.btn_folder.clicked.connect(self.choose_folder)
        bar.addWidget(self.btn_folder)

        self.btn_start = QPushButton('开始检测')
        self.btn_start.setStyleSheet('background:#1f6feb;color:white;font-weight:bold;padding:6px 18px;')
        self.btn_start.clicked.connect(self.start)
        bar.addWidget(self.btn_start)

        self.btn_stop = QPushButton('停止')
        self.btn_stop.clicked.connect(self.stop)
        bar.addWidget(self.btn_stop)

        self.btn_shot = QPushButton('保存当前图')
        self.btn_shot.clicked.connect(self.save_shot)
        bar.addWidget(self.btn_shot)
        bar.addStretch(1)

        wpath = Path(self.detector.weights)
        try:
            n_params = sum(p.numel() for p in self.detector.model.model.parameters())
        except Exception:
            n_params = 0
        if n_params > 5_000_000:
            tag = 'YOLOv8s'
        elif n_params > 1_000_000:
            tag = 'YOLOv8n'
        else:
            tag = '未知模型'
        size_mb = wpath.stat().st_size / 1024.0 / 1024.0
        self.lbl_model = QLabel('模型：%s（%s，%.1f MB）' % (tag, wpath.name, size_mb))
        self.lbl_model.setStyleSheet('color:#666;')
        bar.addWidget(self.lbl_model)
        root.addLayout(bar)

        # 主体：左画面 右面板
        main = QHBoxLayout()
        main.setSpacing(10)

        self.lbl_image = QLabel('等待开始检测')
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(880, 560)
        self.lbl_image.setStyleSheet('background:#111;color:#888;font-size:16px;border:1px solid #333;')
        main.addWidget(self.lbl_image, 1)

        panel = QVBoxLayout()
        panel.setSpacing(8)

        self.lbl_verdict = QLabel('待检')
        self.lbl_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_verdict.setMinimumHeight(96)
        self.lbl_verdict.setStyleSheet(
            'background:#666;color:white;font-size:34px;font-weight:bold;border-radius:6px;')
        panel.addWidget(self.lbl_verdict)

        self.lbl_speed = QLabel('耗时 -- ms')
        self.lbl_speed.setAlignment(Qt.AlignCenter)
        self.lbl_speed.setStyleSheet('color:#333;font-size:12px;')
        panel.addWidget(self.lbl_speed)

        self.lbl_detail = QLabel('面积比 --    缺陷 0    端面 --')
        self.lbl_detail.setAlignment(Qt.AlignCenter)
        self.lbl_detail.setStyleSheet('color:#555;font-size:12px;')
        panel.addWidget(self.lbl_detail)

        gb_par = QGroupBox('判定参数')
        pf = QVBoxLayout(gb_par)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('置信度阈值'))
        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.05, 0.95)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(CONF_DEFAULT)
        row1.addWidget(self.sp_conf)
        pf.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('高置信阈值Tc'))
        self.sp_hard = QDoubleSpinBox()
        self.sp_hard.setRange(0.30, 0.95)
        self.sp_hard.setSingleStep(0.05)
        self.sp_hard.setValue(CONF_HARD_DEFAULT)
        self.sp_hard.setToolTip('任一个缺陷框置信度超过该值，直接判不合格')
        row2.addWidget(self.sp_hard)
        pf.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel('面积比阈值Ta'))
        self.sp_area = QDoubleSpinBox()
        self.sp_area.setRange(0.0, 10.0)
        self.sp_area.setSingleStep(0.1)
        self.sp_area.setDecimals(1)
        self.sp_area.setValue(AREA_RATIO_DEFAULT)
        self.sp_area.setSuffix(' %')
        self.sp_area.setToolTip('缺陷总面积占端面面积的比例超过该值即判不合格')
        row3.addWidget(self.sp_area)
        pf.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel('推理尺寸'))
        self.cmb_imgsz = QComboBox()
        self.cmb_imgsz.addItems(['416', '512', '640', '768'])
        self.cmb_imgsz.setCurrentText(str(IMGSZ_DEFAULT))
        row4.addWidget(self.cmb_imgsz)
        pf.addLayout(row4)
        self.cmb_imgsz.currentTextChanged.connect(self.on_imgsz)
        panel.addWidget(gb_par)

        gb_stat = QGroupBox('统计')
        sf = QVBoxLayout(gb_stat)
        self.lbl_total = QLabel('累计 0 帧')
        self.lbl_okng = QLabel('合格 0    不良 0')
        self.lbl_rate = QLabel('不良率 0.00 %')
        for w in (self.lbl_total, self.lbl_okng, self.lbl_rate):
            w.setStyleSheet('font-size:13px;')
            sf.addWidget(w)
        panel.addWidget(gb_stat)

        gb_log = QGroupBox('最近记录')
        lf = QVBoxLayout(gb_log)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(['时间', '判定', '缺陷数', '耗时ms'])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in (1, 2, 3):
            self.tbl.horizontalHeader().setSectionResizeMode(i, QHeaderView.Fixed)
        self.tbl.setColumnWidth(1, 62)
        self.tbl.setColumnWidth(2, 74)
        self.tbl.setColumnWidth(3, 74)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setMinimumHeight(220)
        lf.addWidget(self.tbl)
        panel.addWidget(gb_log, 1)

        panel_w = QWidget()
        panel_w.setLayout(panel)
        panel_w.setFixedWidth(360)
        main.addWidget(panel_w)
        root.addLayout(main, 1)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(84)
        self.txt_log.setStyleSheet('background:#fafafa;font-size:12px;')
        root.addWidget(self.txt_log)

        self.setCentralWidget(central)
        self.refresh_stats()
        self.log('系统就绪，权重：%s' % self.detector.weights)

    # ---------- 行为 ----------
    def log(self, msg):
        self.txt_log.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, '选择图片文件夹')
        if d:
            self.folder = d
            self.cmb_source.setCurrentText('图片文件夹')
            self.log('选择文件夹：%s' % d)

    def on_source_changed(self, text):
        self.ed_url.setVisible(text == '手机网络摄像头')

    def on_imgsz(self, text):
        try:
            self.detector.imgsz = int(text)
        except Exception:
            pass

    def build_source(self):
        mode = self.cmb_source.currentText()
        if mode == '图片文件夹':
            if not self.folder:
                raise RuntimeError('请先选择图片文件夹')
            return FolderSource(self.folder), 0.0
        if mode == '手机网络摄像头':
            url = self.ed_url.text().strip()
            if not url:
                raise RuntimeError('请先在输入框里填写手机端的视频流地址')
            return CameraSource(url), 0.0
        idx = 1 if mode == '摄像头 1' else 0
        return CameraSource(idx), 0.0

    def start(self):
        if self.worker and self.worker.isRunning():
            self.log('已在运行中')
            return
        try:
            self.source, pace = self.build_source()
        except Exception as e:
            self.log('启动失败：%s' % e)
            return
        self.worker = InferenceWorker(self.source, self.detector, self.sp_conf.value(),
                                      self.sp_hard.value(), self.sp_area.value() / 100.0, pace)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.failed.connect(lambda m: self.log('采集异常：%s' % m))
        self.worker.start()
        self.log('开始检测（%s）' % self.cmb_source.currentText())

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.source:
            self.source.release()
            self.source = None
        self.log('已停止')

    def on_frame(self, ann, jr, fps=None):
        self.show_image(ann)
        speed = '耗时 %.1f ms（推理 %.1f + 绘制 %.1f）' % (
            jr.elapsed_ms, getattr(jr, 'infer_ms', 0.0), getattr(jr, 'draw_ms', 0.0))
        if fps:
            speed += '    帧率 %.1f FPS' % fps
        self.lbl_speed.setText(speed)
        method = getattr(jr, 'face_method', 'fallback')
        if method == 'circle':
            face_txt = '端面 %.0f px²（霍夫圆）' % jr.face_area_px
        else:
            face_txt = '端面 %.0f px²（兜底 %.0f%%）' % (jr.face_area_px, FACE_AREA_FALLBACK * 100)
        self.lbl_detail.setText('面积比 %.2f%%    缺陷 %d    %s'
                                % (jr.area_ratio * 100, jr.n_defects, face_txt))
        self.set_verdict(jr)
        self.maybe_record(ann, jr)

    def show_image(self, bgr):
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        img = QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        pm = QPixmap.fromImage(img).scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(pm)

    def set_verdict(self, jr):
        if jr.is_ng:
            self.lbl_verdict.setText('NG  不良')
            self.lbl_verdict.setStyleSheet(
                'background:#d9363e;color:white;font-size:34px;font-weight:bold;border-radius:6px;')
        else:
            self.lbl_verdict.setText('OK  合格')
            self.lbl_verdict.setStyleSheet(
                'background:#1f9d55;color:white;font-size:34px;font-weight:bold;border-radius:6px;')

    def maybe_record(self, ann, jr):
        now = time.time()
        if now - self.last_record < 0.8 and jr.verdict == self.last_verdict:
            return
        self.last_record = now
        self.last_verdict = jr.verdict
        path = ''
        if jr.is_ng:
            f = SHOT_DIR / ('NG_%s.jpg' % datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3])
            ok, buf = cv2.imencode('.jpg', ann)
            if ok:
                f.write_bytes(buf.tobytes())
                path = str(f)
        detail = {class_cn(k): v for k, v in jr.counts.items()}
        self.db.add(jr.verdict, jr.n_defects, detail, jr.elapsed_ms,
                    source=self.cmb_source.currentText(), image_path=path)
        self.refresh_stats()
        self.refresh_table()

    def refresh_stats(self):
        s = self.db.stats()
        self.lbl_total.setText('累计 %d 帧' % s['total'])
        self.lbl_okng.setText('合格 %d    不良 %d' % (s['ok'], s['ng']))
        self.lbl_rate.setText('不良率 %.2f %%' % s['ng_rate'])

    def refresh_table(self):
        rows = self.db.recent(10)
        self.tbl.setRowCount(len(rows))
        for r, (ts, verdict, n, detail, ms) in enumerate(rows):
            for c, val in enumerate([ts[5:], verdict, str(n), '%.1f' % ms]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 1:
                    item.setForeground(Qt.red if verdict == 'NG' else Qt.darkGreen)
                self.tbl.setItem(r, c, item)

    def save_shot(self):
        pm = self.lbl_image.pixmap()
        if pm is None:
            self.log('当前没有画面')
            return
        f = SHOT_DIR / ('shot_%s.png' % time.strftime('%Y%m%d_%H%M%S'))
        pm.save(str(f))
        self.log('已保存：%s' % f)

    def closeEvent(self, event):
        self.stop()
        self.db.close()
        super().closeEvent(event)
