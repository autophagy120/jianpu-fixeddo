"""预处理方案并排比较：超分 / 插值 / 二值化顺序。

用户提示：三月雨（742x1003，字高 9px）明显最差，怀疑是"先放大再二值化"这条链
把细笔画毁了。这里把几种方案并排跑同一套指标，选最好的接进主管线。

指标都基于同一套几何 + 同一份 OCR 文本：
  行数 / 槽位 / 上点 / 下点 / 配对数 / 数字一致率 / 半截字形占比
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from . import core
from . import merge_octave as MO


PAGES = [
    ("三月雨", r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg",
     r"out\batch\_ocr\三月雨__50b751ebbe319 [原始大小].txt"),
    ("为了你", r"work\为了你唱下去.jpg", r"work\为了你唱下去.ocr.txt"),
]


def load_gray(path):
    return core.load_gray(path)


def v_raw(g):
    return g


def v_cubic3(g):
    return cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def v_lanczos3(g):
    return cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)


def v_sr3(g):
    from . import sr
    if not sr.available():
        return None
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(sr.upscale(bgr, 3), cv2.COLOR_BGR2GRAY)


def v_sr3_sharp(g):
    x = v_sr3(g)
    if x is None:
        return None
    b = cv2.GaussianBlur(x, (0, 0), 1.2)
    return cv2.addWeighted(x, 1.8, b, -0.8, 0)


def v_cubic3_sharp(g):
    x = v_cubic3(g)
    b = cv2.GaussianBlur(x, (0, 0), 1.2)
    return cv2.addWeighted(x, 1.8, b, -0.8, 0)


VARIANTS = [
    ("raw", v_raw),
    ("cubic3", v_cubic3),
    ("cubic3+sharp", v_cubic3_sharp),
    ("lanczos3", v_lanczos3),
    ("sr3", v_sr3),
    ("sr3+sharp", v_sr3_sharp),
]


def run_one(name, path, ocr_path, tag, fn):
    from .glyphnet import GlyphNet
    net = GlyphNet()
    g = load_gray(path)
    out = fn(g)
    if out is None:
        return None
    tmp = os.path.join("work", f"pp_{tag}.png")
    os.makedirs("work", exist_ok=True)
    core.imwrite(tmp, out)
    ocr_txt = open(ocr_path, encoding="utf8").read()
    lines, pairs, st, page = MO.merge_page(tmp, ocr_txt, net)
    from . import align as AL
    slots = AL.ordered_slots(page, net)
    hs = np.array([s.box[3] - s.box[1] for s in slots], dtype=float)
    med = float(np.median(hs)) if hs.size else 1.0
    half = float((hs < 0.72 * med).mean()) if hs.size else 0.0
    return dict(name=name, slots=len(slots), up=sum(1 for s in slots if s.octave > 0),
                dn=sum(1 for s in slots if s.octave < 0), pairs=len(pairs),
                agree=st["agree_rate"], half=half,
                rows=len({int(s.row) for s in slots}))


def main():
    for title, path, ocr in PAGES:
        print(f"\n=== {title} ===")
        print(f"{'方案':<14}{'行':>4}{'槽位':>6}{'上点':>6}{'下点':>6}{'配对':>6}{'一致%':>8}{'半截%':>7}")
        best = None
        for name, fn in VARIANTS:
            try:
                r = run_one(name, path, ocr, title.encode("ascii", "ignore").decode() or "x", fn)
            except Exception as e:
                print(f"{name:<14} 失败: {e}")
                continue
            if not r:
                continue
            print(f"{r['name']:<14}{r['rows']:>4}{r['slots']:>6}{r['up']:>6}{r['dn']:>6}"
                  f"{r['pairs']:>6}{r['agree']*100:>7.1f}{r['half']*100:>7.0f}")
            score = r["agree"] * min(1.0, r["pairs"] / max(1, r["slots"])) - r["half"] * 0.2
            if best is None or score > best[0]:
                best = (score, r["name"])
        if best:
            print(f"  → 该页最优: {best[1]}")


if __name__ == "__main__":
    main()
