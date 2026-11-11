"""诊断/对比出图：带版本号的文件名 + 残墨可视化。

为什么需要：
1. 反复覆盖同一个文件名时，看图软件可能缓存旧图，造成"改了一样看着没变"的假象。
   这里每次输出**新文件名**（带版本号）。
2. "黑色残笔有没有清掉"用眼睛在整页上很难判断。这里直接把
   **原图有墨、但成品仍是黑色**的像素标成绿色，残留一眼可见。
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


def residue_map(orig_path: str, fixed_path: str, out_path: str,
                bands=None) -> dict:
    """把「原图有墨、成品仍为黑色」的像素标绿，输出可视化 + 统计。

    `bands` 限定只统计**谱行带**（(y0,y1) 列表）：歌词本来是黑的、本来就该保留，
    全页统计会把歌词算成"残墨"，得出毫无意义的 60%。
    """
    o = np.array(Image.open(orig_path).convert("L"))
    f_img = Image.open(fixed_path).convert("RGB")
    if o.shape != (f_img.height, f_img.width):
        o = np.array(Image.fromarray(o).resize((f_img.width, f_img.height), Image.LANCZOS))
    arr = np.array(f_img)
    R = arr[:, :, 0].astype(int)
    G = arr[:, :, 1].astype(int)
    B = arr[:, :, 2].astype(int)
    f = arr.mean(axis=2)
    red = (R > 110) & (G < 95) & (B < 95)
    orig_ink = o < 128
    still_dark = (f < 110) & ~red
    resid = orig_ink & still_dark
    mask = np.zeros_like(resid)
    if bands:
        for y0, y1 in bands:
            mask[max(0, int(y0)):min(resid.shape[0], int(y1)), :] = True
        resid = resid & mask
        orig_ink = orig_ink & mask
    n = int(resid.sum())
    tot = int(orig_ink.sum())
    rgb = np.stack([f] * 3, axis=-1)
    rgb[resid] = (0, 190, 60)          # 绿色 = 原图黑墨仍在
    Image.fromarray(rgb.astype(np.uint8)).save(out_path, quality=92)
    return dict(resid_px=n, orig_ink=tot,
                ratio=(n / tot) if tot else 0.0, path=out_path)


def compose(orig_path: str, fixed_path: str, out_path: str, title: str,
            panel_w: int = 0) -> str:
    a = Image.open(orig_path).convert("RGB")
    b = Image.open(fixed_path).convert("RGB")
    if not panel_w:
        panel_w = max(a.width, b.width)

    def fit(im):
        k = panel_w / float(im.width)
        return im.resize((panel_w, max(1, int(round(im.height * k)))), Image.LANCZOS)

    a, b = fit(a), fit(b)
    bar, gap = 44, 20
    H = max(a.height, b.height)
    W = panel_w * 2 + gap
    canvas = Image.new("RGB", (W, bar + H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f = _font(27)
    d.rectangle([0, 0, panel_w, bar], fill=(238, 238, 238))
    d.text((12, 7), f"原谱（黑）　{title}", font=f, fill=(20, 20, 20))
    d.rectangle([panel_w + gap, 0, W, bar], fill=(255, 236, 236))
    d.text((panel_w + gap + 12, 7), "识别结果（红·未转调）", font=f, fill=(150, 0, 0))
    canvas.paste(a, (0, bar))
    canvas.paste(b, (panel_w + gap, bar))
    d.rectangle([panel_w, 0, panel_w + gap - 1, bar + H], fill=(190, 60, 60))
    canvas.save(out_path, quality=92, optimize=True)
    return out_path


JOBS = [
    ("为了你唱下去", r"work\为了你唱下去.jpg", r"out\为了你唱下去_fixed.jpg"),
    ("三月雨", r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg",
     r"out\三月雨_fixed.jpg"),
]


def main(ver: str = "v4"):
    from .glyphnet import GlyphNet
    from . import core as C
    outs = []
    for title, o, fx in JOBS:
        if not (os.path.exists(o) and os.path.exists(fx)):
            print("缺文件：", title)
            continue
        key = "foryou" if "为了你" in title else "sanyue"
        # 谱行带：只在这些 y 区间内统计残墨
        try:
            page = C.analyse(C.load_gray(o), GlyphNet(), path=o)
            bands = [(r.y0, r.y1) for r in page.rows if r.kind == "music"]
        except Exception:
            bands = None
        p1 = compose(o, fx, rf"out\cmp_{ver}_{key}.jpg", title)
        p2 = residue_map(o, fx, rf"out\resid_{ver}_{key}.png", bands)
        print("%-8s 谱行带残墨 %.3f%%  (%d / %d px) -> %s" %
              (title, p2["ratio"] * 100, p2["resid_px"], p2["orig_ink"], p2["path"]))
        outs += [p1, p2["path"]]
    return outs


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v4")
