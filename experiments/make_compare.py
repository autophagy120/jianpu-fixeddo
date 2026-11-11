"""生成「原谱（黑）vs 红字替换后」的上下并排对照图，供主观评判。

参数调优已经扫过：放宽纵向合并阈值只会让数字一致率从 99.6% 掉到 97.3%
（把点并进字形、污染了字形框），而三月雨那 32% 的半截字形一点不动，
所以根因在更上游的分割，不在合并间隙。这里只做对照图。
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core

FONTS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]


def _font(sz: int):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def compose(orig_path: str, fixed_path: str, out_path: str,
            panel_w: int = 0, title: str = "") -> str:
    """左右并排：左＝原谱黑字，右＝识别结果红字（未转调，应完全一致）。

    `panel_w=0` 表示**按原尺寸不缩放**（两侧统一到较宽的那张），
    之前缩到 900 px 导致红色数字糊成一团、没法判断。
    """
    a = Image.open(orig_path).convert("RGB")
    b = Image.open(fixed_path).convert("RGB")
    if not panel_w:
        panel_w = max(a.width, b.width)

    def fit(im):
        k = panel_w / float(im.width)
        return im.resize((panel_w, max(1, int(round(im.height * k)))),
                         Image.LANCZOS) if abs(k - 1) > 1e-3 else im

    a, b = fit(a), fit(b)
    bar = 44
    gap = 20
    H = max(a.height, b.height)
    W = panel_w * 2 + gap
    canvas = Image.new("RGB", (W, bar + H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f = _font(27)

    d.rectangle([0, 0, panel_w, bar], fill=(238, 238, 238))
    d.text((12, 7), f"原谱（黑）　{title}", font=f, fill=(20, 20, 20))
    d.rectangle([panel_w + gap, 0, W, bar], fill=(255, 236, 236))
    d.text((panel_w + gap + 12, 7), "识别结果（红·未转调·应一致）", font=f,
           fill=(150, 0, 0))

    canvas.paste(a, (0, bar))
    canvas.paste(b, (panel_w + gap, bar))
    d.rectangle([panel_w, 0, panel_w + gap - 1, bar + H], fill=(190, 60, 60))
    d.rectangle([0, 0, W - 1, bar + H - 1], outline=(200, 200, 200), width=1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, quality=92, optimize=True)
    return out_path


JOBS = [
    ("为了你唱下去", r"work\为了你唱下去.jpg", r"out\为了你唱下去_fixed.jpg",
     r"out\cmp_为了你唱下去.jpg"),
    ("三月雨", r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg",
     r"out\三月雨_fixed.jpg", r"out\cmp_三月雨.jpg"),
]


def main():
    outs = []
    for title, o, fx, dst in JOBS:
        if not (os.path.exists(o) and os.path.exists(fx)):
            print("缺文件，跳过:", title)
            continue
        p = compose(o, fx, dst, title=title)
        print("写出:", p, " %.1f MB" % (os.path.getsize(p) / 1e6))
        outs.append(p)
    return outs


if __name__ == "__main__":
    main()
