# -*- coding: utf-8 -*-
"""权重下载助手：国内直连 GitHub 会失败，这里走镜像。
用法：
    python download_weights.py yolo26n.pt
    python download_weights.py yolo26s yolo26n-p2 yolo11n
"""
import sys
from pathlib import Path

import requests

MIRRORS = [
    "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/{n}",
    "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/{n}",
    "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/{n}",
]


def download(name: str) -> bool:
    if not name.endswith(".pt"):
        name += ".pt"
    dst = Path(name)
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print("已存在，跳过：%s" % dst.resolve())
        return True
    for m in MIRRORS:
        url = m.format(n=name)
        try:
            print("尝试 %s" % url)
            r = requests.get(url, stream=True, timeout=(20, 180))
            r.raise_for_status()
            data = r.content
            if len(data) < 1_000_000:
                print("  文件过小，换下一个镜像")
                continue
            dst.write_bytes(data)
            print("  成功：%s  %.1f MB" % (dst.resolve(), len(data) / 1024 / 1024))
            return True
        except Exception as e:  # noqa: BLE001
            print("  失败：%s" % str(e)[:120])
    print("全部镜像失败：%s" % name)
    return False


if __name__ == "__main__":
    names = sys.argv[1:] or ["yolo26n.pt"]
    ok = all(download(n) for n in names)
    sys.exit(0 if ok else 1)
