"""在**原始分辨率**上做字模匹配（easytr 的真实做法）：9px 字 vs 9px 渲染模板。

我之前把图放大到 27px 再比对清晰渲染的 27px 模板——模糊 vs 清晰，尺度不对，必错。
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jp import core

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
BASE = r"C:\Windows\Fonts"
FONTS = ["arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "simsun.ttc", "simhei.ttf",
         "msyh.ttc", "calibri.ttf", "verdana.ttf", "tahoma.ttf", "georgia.ttf", "Deng.ttf",
         "BOOKOS.TTF", "consola.ttf", "cour.ttf"]


def render(ch, fp, size):
    try:
        f = ImageFont.truetype(fp, size)
    except Exception:
        return None
    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size, size), ch, fill=255, font=f)
    a = np.array(img)
    ink = (a > 110).astype(np.uint8)
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return None
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_bank(hs):
    bank = []
    for fn in FONTS:
        fp = f"{BASE}\\{fn}"
        for h in hs:
            for fs in range(max(4, h - 3), h + 10):
                for ch in "01234567":
                    m = render(ch, fp, fs)
                    if m is not None and abs(m.shape[0] - h) <= 1:
                        bank.append((ch, m))
    return bank


def iou(a, b, pad=1):
    A, B = a > 0, b > 0
    best = 0.0
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            Bb = np.roll(np.roll(B, dy, axis=0), dx, axis=1)
            h, w = min(A.shape[0], Bb.shape[0]), min(A.shape[1], Bb.shape[1])
            if h <= 0 or w <= 0:
                continue
            aa, cc = A[:h, :w], Bb[:h, :w]
            u = np.logical_or(aa, cc).sum()
            if u:
                best = max(best, float(np.logical_and(aa, cc).sum()) / u)
    return best


def main():
    g = core.load_gray(P)                     # 原始 742x1003
    bw = core.binarize(g)
    clean, _ = core.strip_rules(bw, 9, 7)
    comps = core.components(clean, min_area=4)
    print(f"原始分辨率组件 {len(comps)} 个")

    # 取第一行谱（y 约 130-165）
    line = [c for c in comps if 128 <= c.cy <= 168]
    line.sort(key=lambda c: c.x)
    hs = [c.h for c in line if 6 <= c.h <= 14]
    print(f"该行组件 {len(line)}，其中字高 6-14 的 {len(hs)} 个，高度分布 "
          f"{sorted({h for h in hs})[:12]}")
    glyphs = [c for c in line if 7 <= c.h <= 13 and c.w >= 2]
    print(f"取作字形 {len(glyphs)} 个")

    bank = build_bank(sorted({c.h for c in glyphs}) or [9])
    print(f"模板 {len(bank)} 个")

    seq, conf = [], []
    for c in glyphs:
        p = clean[c.y:c.y + c.h, c.x:c.x + c.w]
        best_c, best_s, second = "?", 0.0, 0.0
        for ch, t in bank:
            s = iou(p, t)
            if s > best_s:
                second, best_s, best_c = best_s, s, ch
            elif s > second:
                second = s
        seq.append(best_c)
        conf.append((round(best_s, 2), round(best_s - second, 2)))
    gt = ("6123" * 20)[:len(seq)]
    print("\n识别:", " ".join(seq))
    print("真值:", " ".join(gt))
    print(f"逐字正确 {sum(1 for a, b in zip(seq, gt) if a == b)}/{len(seq)}")
    print("置信:", conf[:16])


if __name__ == "__main__":
    main()
