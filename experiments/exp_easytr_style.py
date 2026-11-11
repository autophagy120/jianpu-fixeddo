"""easytr 式分类器（确定性、离线）：
   页内聚类 -> 簇中心（多实例平均去噪） -> 与字体渲染的原生模板比对命名 -> 归类
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jp import core, pagetempl

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"

FONTS = ["arial.ttf", "arialbd.ttf", "times.ttf", "simsun.ttc", "simhei.ttf", "msyh.ttc",
         "calibri.ttf", "verdana.ttf", "tahoma.ttf", "georgia.ttf", "Deng.ttf"]
BASE = r"C:\Windows\Fonts"


def render(ch: str, font_path: str, size: int):
    try:
        f = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size, size), ch, fill=255, font=f)
    a = np.array(img)
    ink = (a > 100).astype(np.uint8) * 255
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return None
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def bank_for_size(h_native: int, char_w: int):
    """渲染"高度≈h_native"的数字模板。"""
    bank = []
    for fn in FONTS:
        fp = f"{BASE}\\{fn}"
        for fs in range(max(5, h_native - 3), h_native + 9):
            for ch in "01234567":
                m = render(ch, fp, fs)
                if m is None:
                    continue
                if abs(m.shape[0] - h_native) <= 1:
                    bank.append((ch, m))
    return bank


def iou(a, b, pad=1):
    a = a > 0
    b = b > 0
    best = 0.0
    for dy in (-pad, 0, pad):
        for dx in (-pad, 0, pad):
            bb = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            h = min(a.shape[0], bb.shape[0])
            w = min(a.shape[1], bb.shape[1])
            if h <= 0 or w <= 0:
                continue
            aa, cc = a[:h, :w], bb[:h, :w]
            u = np.logical_or(aa, cc).sum()
            if u:
                best = max(best, np.logical_and(aa, cc).sum() / u)
    return best


def main():
    from jp.glyphnet import GlyphNet

    net = GlyphNet()
    pg = core.analyse(core.load_gray(P), net, path=P)
    H = pg.scale
    row0 = [r for r in pg.rows if r.kind == "music"][0]
    items = []
    for c0 in row0.comps:
        for c in core.split_wide_component(pg.clean, c0, H):
            if c.h >= 0.45 * H and c.area >= 8 and 0.15 <= c.aspect <= 1.2:
                items.append(c)
    items.sort(key=lambda c: c.x)
    patches = [core.patch_of(pg.clean, c, pad=0) for c in items]
    cleaned = [pagetempl.clean_patch(p) for p in patches]
    h_med = int(np.median([p.shape[0] for p in cleaned]))
    print(f"字形 {len(items)}，清洗后高度中位 {h_med} px")

    bank = bank_for_size(h_med, 0)
    print(f"模板 {len(bank)} 个（{len(FONTS)} 字体 × 数字 × 匹配高度）")

    # 逐个直接比对（先不做聚类），看基线
    seq, conf = [], []
    for p in cleaned:
        best_c, best_s = "?", 0.0
        second = 0.0
        for ch, t in bank:
            s = iou(p, t)
            if s > best_s:
                second, best_s, best_c = best_s, s, ch
            elif s > second:
                second = s
        seq.append(best_c)
        conf.append((round(best_s, 2), round(best_s - second, 2)))
    gt = ("6123" * 12)[:len(seq)]
    print("\n识别:", " ".join(seq))
    print("真值:", " ".join(gt))
    ok = sum(1 for a, b in zip(seq, gt) if a == b)
    print(f"逐字正确 {ok}/{len(seq)}")
    print("置信(最佳/间距) 前 12:", conf[:12])


if __name__ == "__main__":
    main()
