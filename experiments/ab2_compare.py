# -*- coding: utf-8 -*-
"""2x3 组图：原图 + 5 种方案，统一"只识别不转调"（输出调 = 原调）。

针对用户反馈：
  - 高低音点还有空间，且**两个点会黏连**（两个小墨块并成一个）
  - 擦除不彻底，有的整行擦不掉，看起来像红字叠在黑字上

五种方案（大胆尝试）：
  1. 原图                      —— 参照
  2. 基准                      —— 现有实现
  3. 擦除·整带白化              —— 把整条谱行带里的墨全部擦掉再重绘
  4. 擦除·后清残余（bold）      —— 画完之后，把谱行带内**非红**的暗像素全清掉，
                                  保证红色下面不可能有黑字
  5. 双点·纵向切分（bold）      —— 点的墨块高度 >= 1.55× 单个点径时，判为两点
  6. 后清残余 + 双点切分        —— 两者合并

输出：out/ab2/<曲子>_2x3.jpg
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core
from . import emit_fixed as EF
from . import merge_octave as MO
from . import music as M

FONTS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]


def _font(sz):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


SONGS = {
    "三月雨": (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg",
             r"out\batch\_ocr\三月雨__50b751ebbe319 [原始大小].txt", "bA"),
    "为了你": (r"work\为了你唱下去.jpg", r"work\为了你唱下去.ocr.txt", "G"),
    "BadApple": (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\Bad Apple！！\371057593e3fc10d8bb [原始大小].jpg",
                 r"out\batch\_ocr\Bad Apple！！__371057593e3fc10d8bb [原始大小].txt", "G"),
}


# ---------------------------------------------------------------- 新算法

def erase_whole_band(gray: np.ndarray, pairs, med_h: float):
    """把整条谱行带的墨全部擦白（不管框在哪），随后靠重绘恢复红色内容。"""
    by_row = {}
    for s, _o in pairs:
        by_row.setdefault(int(s.row), []).append(s)
    for _r, ss in by_row.items():
        y0 = int(min(s.box[1] for s in ss) - 0.35 * med_h)
        y1 = int(max(s.box[3] for s in ss) + 0.35 * med_h)
        x0 = int(min(s.box[0] for s in ss) - 0.8 * med_h)
        x1 = int(max(s.box[2] for s in ss) + 0.8 * med_h)
        gray[max(0, y0):y1, max(0, x0):x1] = 255


def postclean(bgr: np.ndarray, pairs, med_h: float, pad: float = 0.30) -> np.ndarray:
    """**后清残余**：画完之后，把谱行带内所有「非红」的暗像素清成白色。

    这是最激进的擦除方案 —— 它不依赖"槽位框准不准"，只要红色没覆盖到的地方
    出现黑墨，就一律清掉。能彻底消灭"红字叠在黑字上"的观感。
    代价：谱行带内的减时线、连音线、小节线也会被清掉。
    """
    out = bgr.copy()
    B = out[:, :, 0].astype(np.int32)
    G = out[:, :, 1].astype(np.int32)
    R = out[:, :, 2].astype(np.int32)
    red = (R > 110) & (G < 95) & (B < 95)
    dark = (R + G + B) < 330
    by_row = {}
    for s, _o in pairs:
        by_row.setdefault(int(s.row), []).append(s)
    for _r, ss in by_row.items():
        y0 = int(min(s.box[1] for s in ss) - pad * med_h)
        y1 = int(max(s.box[3] for s in ss) + pad * med_h)
        x0 = int(min(s.box[0] for s in ss) - 0.8 * med_h)
        x1 = int(max(s.box[2] for s in ss) + 0.8 * med_h)
        y0, y1 = max(0, y0), min(out.shape[0], y1)
        x0, x1 = max(0, x0), min(out.shape[1], x1)
        m = dark[y0:y1, x0:x1] & (~red[y0:y1, x0:x1])
        sub = out[y0:y1, x0:x1]
        sub[m] = (255, 255, 255)
    return out


def split_stuck_dots(pairs, med_h: float, ratio: float = 1.55) -> int:
    """**双点纵向切分**：两个点黏连时墨块高度约为单点的两倍。

    统计所有八度点墨块的高度中位数作为"单点径"，超过 ratio 倍的判为两个点，
    把该音的点数从 ±1 提升到 ±2。返回被升级的个数。
    """
    hs = []
    for s, _o in pairs:
        for b in s.dot_boxes:
            h = b[3] - b[1]
            if h >= 2:
                hs.append(h)
    if len(hs) < 5:
        return 0
    d1 = float(np.median(hs))
    n = 0
    for s, o in pairs:
        if abs(int(o.octave)) != 1 or not s.dot_boxes:
            continue
        bh = max(b[3] - b[1] for b in s.dot_boxes)
        if bh >= ratio * d1:
            o.octave = 2 if int(o.octave) > 0 else -2
            n += 1
    return n


# ---------------------------------------------------------------- 渲染

def one_variant(song, mode: str, cache):
    orig, ocr_path, key_name = SONGS[song]
    page, pairs, key_pc = cache
    med_h = float(np.median([max(1.0, s.box[3] - s.box[1]) for s, _o in pairs])) if pairs else 1.0
    P = [(s, MO.XO.OcrNote(digit=o.digit, octave=int(o.octave), acc=o.acc,
                           aug_dot=o.aug_dot)) for s, o in pairs]
    two = 0
    if "dot2" in mode:
        two = split_stuck_dots(P, med_h)
    # 渲染：先按 mode 决定是否整带白化（在渲染内部擦除之前做不了，这里用后清代替）
    bgr, st = EF.render_fixed_image(page, P, key_pc, key_pc,   # ref=key → 只识别不转调
                                    mark_missed=True,
                                    erase_mode="band" if "band" in mode else "comp")
    if "post" in mode:
        bgr = postclean(bgr, P, med_h)
    up = sum(1 for _s, o in P if int(o.octave) > 0)
    dn = sum(1 for _s, o in P if int(o.octave) < 0)
    stats = dict(drawn=st.get("drawn", 0), up=up, dn=dn, two=two,
                 unc=st.get("uncovered", 0))
    return bgr, stats


VARIANTS = [
    ("原图", "orig"),
    ("基准", "base"),
    ("擦除·整带白化", "band"),
    ("擦除·后清残余", "post"),
    ("双点·纵向切分", "dot2"),
    ("后清+双点", "post+dot2"),
]


def build(song: str) -> str:
    from .glyphnet import GlyphNet
    orig, ocr_path, key_name = SONGS[song]
    txt = MO.sanitize_ocr(open(ocr_path, encoding="utf8").read())
    lines, pairs, st, page = MO.merge_page(orig, txt, GlyphNet())
    key_pc = M.parse_key(key_name)
    cache = (page, pairs, key_pc)

    tiles, labels = [], []
    img0 = Image.open(orig).convert("RGB")
    tiles.append(img0)
    labels.append(("原图", "只识别不转调 的参照"))
    for name, mode in VARIANTS[1:]:
        bgr, s2 = one_variant(song, mode, cache)
        tmp = os.path.join("work", f"ab2_{song}_{mode.replace('+','_')}.jpg")
        core.imwrite(tmp, bgr)
        tiles.append(Image.open(tmp).convert("RGB"))
        labels.append((name, f"上点{s2['up']} 下点{s2['dn']} 双点{s2['two']} 橙框{s2['unc']}"))

    # 2 行 3 列
    COLS, ROWS = 3, 2
    W = 620
    BAR = 52
    ims = []
    for im in tiles:
        k = W / im.width
        ims.append(im.resize((W, max(1, int(im.height * k))), Image.LANCZOS))
    hmax = max(i.height for i in ims)
    Wt = W * COLS + 8 * (COLS - 1)
    Ht = (BAR + hmax) * ROWS + 8 * (ROWS - 1)
    cv = Image.new("RGB", (Wt, Ht), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    f1, f2 = _font(21), _font(15)
    for idx, (im, (nm, sub)) in enumerate(zip(ims, labels)):
        r, c = divmod(idx, COLS)
        x = c * (W + 8)
        y = r * (BAR + hmax + 8)
        d.rectangle([x, y, x + W, y + BAR], fill=(245, 245, 245))
        d.text((x + 6, y + 4), f"{idx}. {nm}", font=f1, fill=(10, 10, 10))
        d.text((x + 6, y + 28), sub, font=f2, fill=(130, 0, 0))
        cv.paste(im, (x, y + BAR))
        d.rectangle([x, y, x + W - 1, y + BAR + im.height], outline=(190, 190, 190))
    out = os.path.join("out", "ab2", f"{song}_2x3.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out, quality=87, optimize=True)
    print("写出", out, "%.1f MB" % (os.path.getsize(out) / 1e6), flush=True)
    return out


def main():
    outs = []
    for s in SONGS:
        outs.append(build(s))
    return outs


if __name__ == "__main__":
    main()
