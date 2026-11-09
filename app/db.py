# -*- coding: utf-8 -*-
"""检测记录：SQLite 存储与统计查询。"""
import json
import sqlite3
import time
from pathlib import Path

from config import DB_PATH


class InspectionDB:
    def __init__(self, path=DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            'CREATE TABLE IF NOT EXISTS records ('
            ' id INTEGER PRIMARY KEY AUTOINCREMENT,'
            ' ts TEXT NOT NULL,'
            ' source TEXT,'
            ' verdict TEXT,'
            ' n_defects INTEGER,'
            ' detail TEXT,'
            ' image_path TEXT,'
            ' elapsed_ms REAL)'
        )
        self.conn.commit()

    def add(self, verdict, n_defects, counts, elapsed_ms, source='', image_path=''):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute(
            'INSERT INTO records (ts, source, verdict, n_defects, detail, image_path, elapsed_ms)'
            ' VALUES (?,?,?,?,?,?,?)',
            (ts, source, verdict, int(n_defects), json.dumps(counts, ensure_ascii=False), image_path, float(elapsed_ms)),
        )
        self.conn.commit()
        return ts

    def recent(self, limit=10):
        cur = self.conn.execute(
            'SELECT ts, verdict, n_defects, detail, elapsed_ms FROM records ORDER BY id DESC LIMIT ?', (int(limit),)
        )
        return cur.fetchall()

    def stats(self):
        row = self.conn.execute(
            "SELECT COUNT(*),"
            " SUM(CASE WHEN verdict='NG' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN verdict='OK' THEN 1 ELSE 0 END) FROM records"
        ).fetchone()
        total, ng, ok = row[0] or 0, row[1] or 0, row[2] or 0
        rate = (ng / total * 100) if total else 0.0
        return {'total': total, 'ng': ng, 'ok': ok, 'ng_rate': rate}

    def close(self):
        self.conn.close()
