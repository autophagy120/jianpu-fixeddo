"""四种优化方案 × 三首曲子的对照出图。

针对用户反馈的两个问题：
  A) 高低音点识别还有空间，且**两个点**的情况目前被压成 1 个（`_norm_oct` 把 ±2 夹到 ±1）
  B) 原字符擦除有时不彻底，看起来像红字叠在黑字上

四种方案：
  1. 基准      —— 现在的实现
  2. 擦除加强  —— erase_mode='band'，按"音符格子"整块擦（含框外残留）
  3. 双点支持  —— 允许 ±2 个八度点（放开 _norm_oct 的 clamp + 放宽点判据）
  4. 两者合并  —— 擦除加强 + 双点支持

输出：out/ab/<曲子>_4panel.jpg（四格并排，标注方案名）
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core
from . import emit_fixed as EF
from . import merge_octave as MO
from . import music as M
from . import xiaomi_ocr as XO

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

VARIANTS = [
    ("基准", dict(erase="comp", dots2=False)),
    ("擦除加强", dict(erase="band", dots2=False)),
    ("双点支持", dict(erase="comp", dots2=True)),
    ("擦除+双点", dict(erase="band", dots2=True)),
]

# 双点模式：放开 ±2 的夹取，并放宽点判据（尺寸上限与宽高比）
_ORIG_NORM = MO._norm_oct
_ORIG_ISDOT = core._is_dot_like


def _norm2(v):
    try:
        return max(-2, min(2, int(round(float(v or 0)))))
    except Exception:
        return 0


def _isdot2(m, scale, clean=None):
    H = max(1.0, float(scale))
    if m.h <= 0 or m.w <= 0 or m.h < 2 or m.w < 2:
        return False
    if m.h > 0.62 * H or m.w > 0.62 * H:      # 放宽尺寸上限
        return False
    asp = m.w / float(m.h)
    if asp >= 2.2 or asp < 0.30:              # 放宽扁长判据
        return False
    if m.area / float(m.w * m.h) < 0.45:
        return False
    return True


def render_one(song: str, variant: dict) -> str:
    orig, ocr_path, key_name = SONGS[song]
    txt = MO.sanitize_ocr(open(ocr_path, encoding="utf8").read())
    from .glyphnet import GlyphNet
    if variant["dots2"]:
        MO._norm_oct = _norm2
        core._is_dot_like = _isdot2
    else:
        MO._norm_oct = _ORIG_NORM
        core._is_dot_like = _ORIG_ISDOT
    try:
        lines, pairs, st, page = MO.merge_page(orig, txt, GlyphNet())
        key_pc = M.parse_key(key_name)
        alln = [n for l in lines for n in l if n.digit != "0"]
        pitches = [M.note_to_abs(int(n.digit), n.acc, int(n.octave), key_pc) for n in alln]
        ref_pc, _r = M.pick_reference(pitches, 3) if pitches else (key_pc, "")
        bgr, stt = EF.render_fixed_image(page, pairs, key_pc, ref_pc,
                                         erase_mode=variant["erase"])
        upd = sum(1 for s, o in pairs if int(o.octave) > 0)
        dwn = sum(1 for s, o in pairs if int(o.octave) < 0)
        two = sum(1 for s, o in pairs if abs(int(o.octave)) >= 2)
        tmp = os.path.join("work", f"ab_{song}_{variant['erase']}_{int(variant['dots2'])}.jpg")
        core.imwrite(tmp, bgr)
        return tmp, dict(drawn=stt.get("drawn", 0), up=upd, dn=dwn, two=two,
                         uncovered=stt.get("uncovered", 0))
    finally:
        MO._norm_oct = _ORIG_NORM
        core._is_dot_like = _ORIG_ISDOT


def build_panel(song: str, tiles, notes) -> str:
    W = 760
    ims = []
    for p in tiles:
        im = Image.open(p).convert("RGB")
        k = W / im.width
        ims.append(im.resize((W, max(1, int(im.height * k))), Image.LANCZOS))
    H = max(i.height for i in ims)
    BAR, GAP = 66, 10
    TW = W * len(ims) + GAP * (len(ims) - 1)
    cv = Image.new("RGB", (TW, BAR + H), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    f1, f2 = _font(22), _font(17)
    for i, (im, (name, _v), nd) in enumerate(zip(ims, VARIANTS, notes)):
        x = i * (W + GAP)
        d.rectangle([x, 0, x + W, BAR], fill=(244, 244, 244))
        d.text((x + 8, 6), f"{i+1}. {name}", font=f1, fill=(15, 15, 15))
        d.text((x + 8, 36), f"上点{nd['up']} 下点{nd['dn']} 双点{nd['two']} 橙框{nd['uncovered']}",
               font=f2, fill=(120, 0, 0))
        cv.paste(im, (x, BAR))
        if i:
            d.rectangle([x - GAP, 0, x - 1, BAR + H], fill=(180, 60, 60))
    out = os.path.join("out", "ab", f"{song}_4panel.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out, quality=88, optimize=True)
    return out


def main(only=None):
    outs = []
    for song in SONGS:
        if only and song not in only:
            continue
        tiles, notes = [], []
        for name, v in VARIANTS:
            t, nd = render_one(song, v)
            tiles.append(t)
            notes.append(nd)
            print(f"{song:10s} {name:8s} drawn={nd['drawn']} up={nd['up']} dn={nd['dn']} "
                  f"2dot={nd['two']} uncovered={nd['uncovered']}", flush=True)
        outs.append(build_panel(song, tiles, notes))
    return outs


if __name__ == "__main__":
    main()
