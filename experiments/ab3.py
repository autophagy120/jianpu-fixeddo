# -*- coding: utf-8 -*-
"""行锚定配对 + 原生分辨率 2x3 组图。

诊断结论：全局 DP 把 OCR token 跨行错配，导致整行只配上 1 个音
（BadApple 实测 [1,40,29,24,28,28,20,1,1,1,6,33]）。
本脚本改为**每行独立配对**：先用音符数把几何行与 OCR 谱行一一对齐，
再在行内配对，某行缺 token 只影响该行，不会把别的行的 token 抢走。

输出：out/ab3/<曲子>_2x3.jpg，每格是**原生分辨率**的区域裁剪（不缩放），
六格分别是：原图 / 旧(全局DP) / 新(行锚定) / 新+后清 / 新+双点 / 新+后清+双点
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import align as AL
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


# ---------------------------------------------------------------- 行锚定配对

def _align_counts(a, b):
    """把 b 的每个元素对齐到 a 的下标（按数值接近做单调对齐）。"""
    m, n = len(b), len(a)
    if m == 0 or n == 0:
        return [None] * m
    NEG = -1e18
    dp = np.full((m + 1, n + 1), NEG)
    bt = np.zeros((m + 1, n + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] - 2.0
        bt[0, j] = 2
    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] - 2.0
        bt[i, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best = dp[i - 1, j - 1] + 6.0 - min(3.0, abs(b[i - 1] - a[j - 1]))
            mv = 0
            v = dp[i - 1, j] - 2.0
            if v > best:
                best, mv = v, 1
            v = dp[i, j - 1] - 2.0
            if v > best:
                best, mv = v, 2
            dp[i, j] = best
            bt[i, j] = mv
    out = [None] * m
    i, j = m, n
    while i > 0 or j > 0:
        mv = bt[i, j]
        if i > 0 and j > 0 and mv == 0:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and mv == 1:
            i -= 1
        elif j > 0 and mv == 2:
            j -= 1
        else:
            i, j = i - 1, j - 1
    return out


def pair_by_rows(page, slots, raw_lines):
    by_row = {}
    for s in slots:
        by_row.setdefault(int(s.row), []).append(s)
    grows = [sorted(v, key=lambda t: t.x) for _k, v in sorted(by_row.items())]
    ocr = []
    for l in raw_lines:
        p = MO.XO.parse_line(l)
        if not p.is_music:
            continue
        ocr.append([MO.XO.OcrNote(digit=n.digit, octave=max(-1, min(1, int(n.octave or 0))),
                                  acc=n.acc, aug_dot=n.aug_dot) for n in p.notes])
    if not grows or not ocr:
        return []
    mp = _align_counts([len(x) for x in ocr], [len(g) for g in grows])
    pairs = []
    for gi, oi in enumerate(mp):
        if oi is None:
            continue
        g, o = grows[gi], ocr[oi]
        if len(g) == len(o):
            pairs.extend(zip(g, o))
        else:
            a = AL.align_streams([n.digit for n in o], [s.digit for s in g])
            for k, j in enumerate(a):
                if j is not None:
                    pairs.append((g[j], o[k]))
    pairs.sort(key=lambda p: (int(p[0].row), p[0].x))
    return pairs


def split_stuck_dots(pairs, ratio=1.55):
    hs = [b[3] - b[1] for s, _o in pairs for b in s.dot_boxes if b[3] - b[1] >= 2]
    if len(hs) < 5:
        return 0
    d1 = float(np.median(hs))
    n = 0
    for s, o in pairs:
        if abs(int(o.octave)) != 1 or not s.dot_boxes:
            continue
        if max(b[3] - b[1] for b in s.dot_boxes) >= ratio * d1:
            o.octave = 2 if int(o.octave) > 0 else -2
            n += 1
    return n


def postclean(bgr, pairs, med_h, pad=0.30):
    out = bgr.copy()
    B, G, R = out[:, :, 0].astype(np.int32), out[:, :, 1].astype(np.int32), out[:, :, 2].astype(np.int32)
    red = (R > 110) & (G < 95) & (B < 95)
    dark = (R + G + B) < 330
    by_row = {}
    for s, _o in pairs:
        by_row.setdefault(int(s.row), []).append(s)
    for _r, ss in by_row.items():
        y0 = max(0, int(min(s.box[1] for s in ss) - pad * med_h))
        y1 = min(out.shape[0], int(max(s.box[3] for s in ss) + pad * med_h))
        x0 = max(0, int(min(s.box[0] for s in ss) - 0.8 * med_h))
        x1 = min(out.shape[1], int(max(s.box[2] for s in ss) + 0.8 * med_h))
        sub = out[y0:y1, x0:x1]
        m = dark[y0:y1, x0:x1] & (~red[y0:y1, x0:x1])
        sub[m] = (255, 255, 255)
    return out


# ---------------------------------------------------------------- 组图

def build(song: str, win_frac: float = 0.34) -> str:
    from .glyphnet import GlyphNet
    orig, ocr_path, key_name = SONGS[song]
    txt = MO.sanitize_ocr(open(ocr_path, encoding="utf8").read())
    net = GlyphNet()
    page = core.analyse(core.load_gray(orig), net, path=orig)
    slots = AL.ordered_slots(page, net)
    raw = [l for l in txt.splitlines() if MO.XO.parse_line(l).is_music]
    key_pc = M.parse_key(key_name)
    med_h = float(np.median([s.box[3] - s.box[1] for s in slots])) if slots else 20.0

    # 旧策略（全局 DP）
    old_lines, old_pairs, _st, _p = MO.merge_page(orig, txt, net)
    new_pairs = pair_by_rows(page, slots, raw)

    def render(prs, do2=False, do_post=False):
        P = [(s, MO.XO.OcrNote(digit=o.digit, octave=int(o.octave), acc=o.acc,
                               aug_dot=o.aug_dot)) for s, o in prs]
        two = split_stuck_dots(P) if do2 else 0
        bgr, st = EF.render_fixed_image(page, P, key_pc, key_pc, mark_missed=True,
                                        erase_mode="comp")
        if do_post:
            bgr = postclean(bgr, P, med_h)
        up = sum(1 for _s, o in P if int(o.octave) > 0)
        dn = sum(1 for _s, o in P if int(o.octave) < 0)
        return bgr, dict(n=len(P), up=up, dn=dn, two=two, unc=st.get("uncovered", 0))

    variants = []
    b, s0 = render(old_pairs); variants.append(("旧·全局DP", s0, b))
    b, s1 = render(new_pairs); variants.append(("新·行锚定", s1, b))
    b, s2 = render(new_pairs, do_post=True); variants.append(("新+后清残余", s2, b))
    b, s3 = render(new_pairs, do2=True); variants.append(("新+双点切分", s3, b))
    b, s4 = render(new_pairs, do2=True, do_post=True); variants.append(("新+双点+后清", s4, b))

    # 裁剪窗口：取问题最多的一段（旧策略里配对数最少的那一行附近）
    per = {}
    for s, o in old_pairs:
        per.setdefault(int(s.row), 0)
        per[int(s.row)] += 1
    rows = sorted({int(s.row) for s in slots})
    bad = min(per, key=lambda r: per[r]) if per else rows[0]
    ys = [int(min(s.box[1] for s in slots if int(s.row) == r)) for r in rows if r in per] or [0]
    y_focus = int(min(s.box[1] for s in slots if int(s.row) == bad)) if bad in rows else ys[0]
    H = int(page.gray.shape[0] * win_frac)
    y0 = max(0, y_focus - int(0.25 * H))
    y1 = min(page.gray.shape[0], y0 + H)

    tiles = []
    o_img = Image.open(orig).convert("RGB")
    k = page.gray.shape[1] / float(o_img.width)
    tiles.append(("原图", f"区域 y={y0}..{y1}", o_img.crop(
        (0, int(y0 / k), o_img.width, int(y1 / k))).resize(
        (page.gray.shape[1], y1 - y0), Image.LANCZOS)))
    for name, st, bgr in variants:
        tiles.append((name, f"音{st['n']} 上点{st['up']} 下点{st['dn']} 双{st['two']} 橙{st['unc']}",
                      Image.fromarray(bgr[y0:y1, :, ::-1])))

    COLS, ROWS = 3, 2
    Wp = max(t[2].width for t in tiles)
    BAR = 54
    hmax = max(t[2].height for t in tiles)
    cv = Image.new("RGB", (Wp * COLS + 10 * (COLS - 1), (BAR + hmax) * ROWS + 10), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    f1, f2 = _font(26), _font(19)
    for idx, (nm, sub, im) in enumerate(tiles):
        r, c = divmod(idx, COLS)
        x, y = c * (Wp + 10), r * (BAR + hmax + 10)
        d.rectangle([x, y, x + Wp, y + BAR], fill=(245, 245, 245))
        d.text((x + 8, y + 3), f"{idx}. {nm}", font=f1, fill=(10, 10, 10))
        d.text((x + 8, y + 30), sub, font=f2, fill=(130, 0, 0))
        cv.paste(im, (x, y + BAR))
        d.rectangle([x, y, x + Wp - 1, y + BAR + im.height], outline=(185, 185, 185))
    out = os.path.join("out", "ab3", f"{song}_2x3.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out, quality=93, optimize=True)
    print("写出", out, " 像素 %dx%d  %.1f MB" % (cv.width, cv.height,
                                                os.path.getsize(out) / 1e6), flush=True)
    return out


def main():
    return [build(s) for s in SONGS]


if __name__ == "__main__":
    main()
