# -*- coding: utf-8 -*-
"""削横线后按音符单元数八度点（重写高低音识别）。

旧实现的问题：它直接拿"连通块"判点。八度点一旦和减时线（横杠）粘连，
连通块就变成宽扁块，被当成减时线否掉 —— 十六分音符连打、点紧贴双横线的
行基本全丢（用户给的那条 `6 · 0656 6 · 0656` 就是这种）。

新算法的顺序反过来：
  1. 先把谱行带里的**长横线**用形态学开运算整条提取出来，从墨迹里减掉；
  2. 剩下的是不含横线的墨（点、竖笔画碎块）；
  3. 以每个音的 x 单元为界（到相邻音中点），在数字**下方 / 上方**的窗口里
     数小墨块的个数 —— 横向排开的多个点也能数对；
  4. 上/下各最多 3 个，超过忽略（用户说三点以上是极特殊情况）。

输出对照图：out/dots2/<曲子>.jpg（上：旧算法；下：新算法）
"""
from __future__ import annotations

import os

import cv2
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


def build_residual(page: core.Page, med_h: float) -> np.ndarray:
    """去掉长横线后的墨迹图（ink=255）。"""
    bw = (page.gray < 128).astype(np.uint8) * 255
    kw = max(9, int(1.1 * med_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    resid = cv2.subtract(bw, horiz)
    return resid


def count_dots(resid, x_lo, x_hi, y_lo, y_hi, med_h) -> int:
    H, W = resid.shape
    x0, x1 = max(0, int(x_lo)), min(W, int(x_hi))
    y0, y1 = max(0, int(y_lo)), min(H, int(y_hi))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0
    sub = resid[y0:y1, x0:x1]
    if sub.max() == 0:
        return 0
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats((sub > 0).astype(np.uint8), 8)
    cnt = 0
    maxhw = 0.42 * med_h
    for i in range(1, n):
        w, h, a = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]), int(stats[i, cv2.CC_STAT_AREA])
        if a < 2 or h < 2 or w < 2:
            continue
        if h > maxhw or w > maxhw:
            continue
        if w > 2.4 * h or h > 2.4 * w:
            continue
        if a / float(w * h) < 0.42:
            continue
        cnt += 1
    return cnt


def octaves_from_cells(page, pairs, med_h):
    """用新算法给每个音重算八度，返回 {id(slot): octave}。"""
    resid = build_residual(page, med_h)
    by_row = {}
    for s, _o in pairs:
        by_row.setdefault(int(s.row), []).append(s)
    result = {}
    for _r, ss in by_row.items():
        ss.sort(key=lambda s: s.box[0])
        for i, s in enumerate(ss):
            x0, y0, x1, y1 = s.box
            gh = max(4.0, float(y1 - y0))
            lo = ss[i - 1].box[2] if i > 0 else x0 - 0.5 * gh
            hi = ss[i + 1].box[0] if i + 1 < len(ss) else x1 + 0.5 * gh
            cx_lo = max(lo, x0 - 0.55 * gh)
            cx_hi = min(hi, x1 + 0.55 * gh)
            up = count_dots(resid, cx_lo, cx_hi, y0 - 0.42 * gh, y0 - 0.03 * gh, med_h)
            dn = count_dots(resid, cx_lo, cx_hi, y1 + 0.03 * gh, y1 + 0.42 * gh, med_h)
            up, dn = min(up, 3), min(dn, 3)
            if up and not dn:
                result[id(s)] = up
            elif dn and not up:
                result[id(s)] = -dn
            elif up and dn:
                result[id(s)] = up if up >= dn else -dn
            else:
                result[id(s)] = 0
    return result


def render_pair(page, base_pairs, key_pc, med_h, use_new: bool):
    P = [(s, MO.XO.OcrNote(digit=o.digit, octave=int(o.octave), acc=o.acc, aug_dot=o.aug_dot))
         for s, o in base_pairs]
    if use_new:
        ov = octaves_from_cells(page, P, med_h)
        for s, o in P:
            o.octave = ov.get(id(s), 0)
    bgr, st = EF.render_fixed_image(page, P, key_pc, key_pc, mark_missed=False, erase_mode="comp")
    up = sum(1 for _s, o in P if int(o.octave) > 0)
    dn = sum(1 for _s, o in P if int(o.octave) < 0)
    two = sum(1 for _s, o in P if abs(int(o.octave)) >= 2)
    three = sum(1 for _s, o in P if abs(int(o.octave)) >= 3)
    return bgr, dict(n=len(P), up=up, dn=dn, two=two, three=three)


def build(song, win_frac=0.42):
    from .glyphnet import GlyphNet
    orig, ocr_path, key_name = SONGS[song]
    txt = MO.sanitize_ocr(open(ocr_path, encoding="utf8").read())
    net = GlyphNet()
    page = core.analyse(core.load_gray(orig), net, path=orig)
    slots = AL.ordered_slots(page, net)
    lines, pairs, st, page2 = MO.merge_page(orig, txt, net)
    page = page2
    key_pc = M.parse_key(key_name)
    med_h = float(np.median([s.box[3] - s.box[1] for s, _o in pairs])) if pairs else 20.0

    old_bgr, s_old = render_pair(page, pairs, key_pc, med_h, use_new=False)
    new_bgr, s_new = render_pair(page, pairs, key_pc, med_h, use_new=True)

    # 裁剪：选旧算法带点数最多的行（点最密的行最能看出差别）
    per = {}
    for s, o in pairs:
        if int(o.octave) != 0:
            per[int(s.row)] = per.get(int(s.row), 0) + 1
    rows = sorted({int(s.row) for s, _o in pairs})
    focus = max(per, key=lambda r: per[r]) if per else rows[0]
    ys = [int(min(t.box[1] for t in slots if int(t.row) == r)) for r in rows]
    y_focus = int(min(t.box[1] for t in slots if int(t.row) == focus))
    H = int(page.gray.shape[0] * win_frac)
    y0 = max(0, y_focus - int(0.15 * H))
    y1 = min(page.gray.shape[0], y0 + H)

    tiles = []
    o_img = Image.open(orig).convert("RGB")
    k = page.gray.shape[1] / float(o_img.width)
    tiles.append(("原图", "参照",
                  o_img.crop((0, int(y0 / k), o_img.width, int(y1 / k)))
                  .resize((page.gray.shape[1], y1 - y0), Image.LANCZOS)))
    for nm, stt, bgr in [("旧算法", s_old, old_bgr), ("新算法·削线数点", s_new, new_bgr)]:
        tiles.append((nm, f"上点{stt['up']} 下点{stt['dn']} 双点{stt['two']} 三点{stt['three']}",
                      Image.fromarray(bgr[y0:y1, :, ::-1])))

    Wp = max(t[2].width for t in tiles)
    BAR = 54
    hmax = max(t[2].height for t in tiles)
    cv = Image.new("RGB", (Wp, (BAR + hmax) * len(tiles) + 10), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    f1, f2 = _font(26), _font(19)
    for i, (nm, sub, im) in enumerate(tiles):
        y = i * (BAR + hmax + 10)
        d.rectangle([0, y, Wp, y + BAR], fill=(245, 245, 245))
        d.text((8, y + 3), f"{i}. {nm}", font=f1, fill=(10, 10, 10))
        d.text((8, y + 30), sub, font=f2, fill=(130, 0, 0))
        cv.paste(im, (0, y + BAR))
        d.rectangle([0, y, Wp - 1, y + BAR + im.height], outline=(185, 185, 185))
    out = os.path.join("out", "dots2", f"{song}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out, quality=93, optimize=True)
    print("写出", out, "%dx%d 旧:上%d下%d 双%d | 新:上%d下%d 双%d 三%d" % (
        cv.width, cv.height, s_old["up"], s_old["dn"], s_old["two"],
        s_new["up"], s_new["dn"], s_new["two"], s_new["three"]), flush=True)
    return out


def main():
    return [build(s) for s in SONGS]


if __name__ == "__main__":
    main()
