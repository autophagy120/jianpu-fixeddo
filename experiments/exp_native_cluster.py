"""原生分辨率聚类 + VLM 命名（最终配方）。

- 聚类在 **9px 原始分辨率**上做（不放大），簇内 IoU 0.94-1.0，非常紧；
- 簇代表**放大 10 倍**拼图给 VLM 认（它看得清）；
- 归类 = 按簇归属（确定性），不再做模糊的最近邻。
"""
from __future__ import annotations

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core, pagetempl

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"


def norm9(p, size=20):
    a = p if p.dtype == np.uint8 else p.astype(np.uint8)
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return np.zeros((size, size), np.uint8)
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    s = (size - 4) / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    res = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
    can = np.zeros((size, size), np.uint8)
    can[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = res
    return (can > 100).astype(np.uint8)


def main():
    g = core.load_gray(P)
    bw = core.binarize(g)
    clean, _ = core.strip_rules(bw, 9, 7)
    comps = core.components(clean, min_area=4)
    glyphs = [c for c in comps if 7 <= c.h <= 13 and 2 <= c.w <= 12 and 0.2 <= c.aspect <= 1.3]
    glyphs.sort(key=lambda c: (c.y, c.x))
    print(f"全页字形 {len(glyphs)} 个（原始 9px 尺度）")
    patches = [clean[c.y:c.y + c.h, c.x:c.x + c.w] for c in glyphs]
    masks = [norm9(p) for p in patches]

    # ---- 原生聚类
    clusters = []
    for i, m in enumerate(masks):
        hit = None
        for c in clusters:
            cm = masks[c[0]]
            u = np.logical_or(m, cm).sum()
            s = np.logical_and(m, cm).sum() / u if u else 0
            if s >= 0.80:
                hit = c
                break
        if hit is None:
            clusters.append([i])
        else:
            hit.append(i)
    clusters.sort(key=lambda c: -len(c))
    clusters = [c for c in clusters if len(c) >= 3][:9]
    print("簇大小:", [len(c) for c in clusters])

    # ---- 拼图（每簇代表放大 10 倍）+ VLM 命名
    tile, cols = 110, 5
    rows = int(np.ceil(len(clusters) / cols))
    canvas = np.full((rows * (tile + 40), cols * (tile + 16), 3), 255, np.uint8)
    for i, cl in enumerate(clusters):
        p = patches[cl[0]]
        big = cv2.resize(255 - (p * 255 if p.max() <= 1 else p), (tile, tile),
                         interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        y0, x0 = r * (tile + 40) + 32, c * (tile + 16) + 8
        canvas[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + tile + 1, y0 + tile + 1), (180, 180, 180), 1)
        cv2.putText(canvas, f"#{i}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    cv2.imwrite("out/native_montage.png", canvas)
    labels = pagetempl.ask_labels(canvas)
    print("VLM 命名:", labels)

    # ---- 归类 + 校验第一行
    lab_of = {}
    for i, cl in enumerate(clusters):
        lab = labels.get(str(i), "?")
        for idx in cl:
            lab_of[idx] = lab
    line_y = [c.cy for c in glyphs if 128 <= c.cy <= 168]
    print(f"第一行字形 {len(line_y)} 个")
    first = [lab_of.get(i, "?") for i, c in enumerate(glyphs) if 128 <= c.cy <= 168]
    print("识别:", " ".join(first))
    print("真值: ( 6 1 2 3 6 1 2 3 6 1 2 3 6 1 2 3 6 1 2 3")


if __name__ == "__main__":
    main()
