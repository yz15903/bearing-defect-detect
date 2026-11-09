# -*- coding: utf-8 -*-
"""把一堆「图片 + YOLO txt 标签」按 7:2:1 划分到 datasets/bearing 下，并做泄漏检查。

按工件分组划分：文件名形如 20261110_001_crack_a1.jpg，
默认取前两段（20261110_001）作为同一个工件，同一工件的所有图片只会落在同一个集合里。
如果你的命名规则不同，用 --group-parts 调整。

用法：
    python split_dataset.py --src raw_images --labels raw_labels
    python split_dataset.py --src raw_images --labels raw_labels --group-parts 1
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="原始图片目录")
    p.add_argument("--labels", default=None, help="原始标签(txt)目录，默认与 --src 同级同名 labels")
    p.add_argument("--dst", default="datasets/bearing", help="目标数据集目录")
    p.add_argument("--group-parts", type=int, default=2, help="用文件名前几段作为工件 ID")
    p.add_argument("--ratios", default="0.7,0.2,0.1", help="train,val,test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="复制而不是移动（原文件保留，更安全）")
    return p.parse_args()


def group_key(stem, parts):
    seg = stem.split("_")
    return "_".join(seg[:parts]) if parts > 0 else stem


def main():
    a = parse_args()
    src = Path(a.src)
    lab = Path(a.labels) if a.labels else src.parent / "labels"
    dst = Path(a.dst)

    if not src.is_dir():
        raise SystemExit("图片目录不存在：%s" % src)
    if not lab.is_dir():
        print("警告：标签目录不存在 %s，将按「全部是正常样本(空标签)」处理" % lab)

    groups = defaultdict(list)
    for img in sorted(src.iterdir()):
        if img.suffix.lower() in IMG_EXT:
            groups[group_key(img.stem, a.group_parts)].append(img)

    keys = sorted(groups)
    random.seed(a.seed)
    random.shuffle(keys)

    r = [float(x) for x in a.ratios.split(",")]
    assert len(r) == 3 and abs(sum(r) - 1) < 1e-6, "ratios 必须和为 1，例如 0.7,0.2,0.1"
    n = len(keys)
    n_tr = int(n * r[0])
    n_va = int(n * r[1])
    splits = {
        "train": keys[:n_tr],
        "val": keys[n_tr:n_tr + n_va],
        "test": keys[n_tr + n_va:],
    }

    stats = {}
    for split, gks in splits.items():
        idir = dst / "images" / split
        ldir = dst / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        cnt = 0
        for gk in gks:
            for img in groups[gk]:
                target = idir / img.name
                if target.exists():
                    target.unlink()
                if a.copy:
                    shutil.copy2(img, target)
                else:
                    shutil.move(str(img), str(target))
                txt = lab / (img.stem + ".txt")
                out = ldir / (img.stem + ".txt")
                if txt.exists():
                    shutil.copy2(txt, out)
                else:
                    out.write_text("", encoding="utf-8")  # 无缺陷样本 = 空标签
                cnt += 1
        stats[split] = (len(gks), cnt)

    print("=" * 50)
    print("按工件分组划分结果（工件数 / 图片数）")
    for k, v in stats.items():
        print("  %-6s %4d / %4d" % (k, v[0], v[1]))

    # 泄漏检查
    tr = {p.stem for p in (dst / "images" / "train").iterdir() if p.suffix.lower() in IMG_EXT}
    va = {p.stem for p in (dst / "images" / "val").iterdir() if p.suffix.lower() in IMG_EXT}
    te = {p.stem for p in (dst / "images" / "test").iterdir() if p.suffix.lower() in IMG_EXT}
    if tr & va or tr & te or va & te:
        print("!! 存在跨集合重复文件名，请检查命名规则")
    else:
        print("泄漏检查通过：三个集合无重复图片")

    yml = dst / "data.yaml"
    if not yml.exists():
        yml.write_text(
            "path: %s\n" % dst.resolve().as_posix().replace("\\", "/") +
            "train: images/train\nval: images/val\ntest: images/test\n\n"
            "names:\n  0: scratch\n  1: rust\n  2: bruise\n",
            encoding="utf-8",
        )
        print("已生成 data.yaml：%s（类别名请按实际修改）" % yml)
    else:
        print("data.yaml 已存在，未覆盖：%s" % yml)


if __name__ == "__main__":
    main()
