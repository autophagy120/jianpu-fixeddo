# -*- coding: utf-8 -*-
"""八度点识别 v3：距离变换峰值计数（分水岭思路）。

参考：Deep Watershed Detector (arXiv:1805.10548) —— 针对"高分辨率、大量极小目标"
的分割方法。核心不是"先把线削掉再数"，而是在**距离变换**上判定对象：
  - 减时线是细线（厚度 2~3 px），距离变换峰值 ≈ 1~1.5；
  - 八度点是圆点（直径 0.2~0.35 字高），距离变换峰值 ≈ 直径/2，明显更大。
所以在「数字上方/下方窗口」里数**距离变换的局部极大值**，就能把点从粘连的
横线里分出来 —— 不需要切除，也不会像形态学削线那样把点一起削掉。

旧算法（削横线）在 BadApple 上把下点从 69 打到 14，就是这个原因。

输出 out/dots3/<曲子>.jpg：原图 / 旧算法 / 新算法(v3) 三格（原生分辨率裁剪）
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


def distance_peaks(page: core.Page, med_h: float) -> np.ndarray:
    """返回峰值图（>0 为局部极大点，值=该处半径）。"""
    bw = (page.gray < 128).astype(np.uint8)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    k = max(3, int(round(0.45 * med_h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(dist, kernel)
    # 峰值：等于局部极大且半径足够（排除 1~1.5px 的细线）
    thr = max(1.6, 0.055 * med_h)
    peaks = ((dist >= dil - 1e-6) & (dist >= thr)).astype(np.uint8)
    return peaks


def count_peaks(peaks, x_lo, x_hi, y_lo, y_hi, med_h) -> int:
    H, W = peaks.shape
    x0, x1 = max(0, int(x_lo)), min(W, int(x_hi))
    y0, y1 = max(0, int(y_lo)), min(H, int(y_hi))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0
    sub = peaks[y0:y1, x0:x1]
    if sub.max() == 0:
        return 0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(sub, 8)
    return max(0, n - 1)


def octaves_v3(page, pairs, med_h):
    pk = distance_peaks(page, med_h)
    by_row = {}
    for s, _o in pairs:
        by_row.setdefault(int(s.row), []).append(s)
    res = {}
    for _r, ss in by_row.items():
        ss.sort(key=lambda s: s.box[0])
        for i, s in enumerate(ss):
            x0, y0, x1, y1 = s.box
            gh = max(4.0, float(y1 - y0))
            lo = ss[i - 1].box[2] if i > 0 else x0 - 0.5 * gh
            hi = ss[i + 1].box[0] if i + 1 < len(ss) else x1 + 0.5 * gh
            cx_lo = max(lo, x0 - 0.55 * gh)
            cx_hi = min(hi, x1 + 0.55 * gh)
            up = min(3, count_peaks(pk, cx_lo, cx_hi, y0 - 0.42 * gh, y0 - 0.02 * gh, med_h))
            dn = min(3, count_peaks(pk, cx_lo, cx_hi, y1 + 0.02 * gh, y1 + 0.42 * gh, med_h))
            if up and not dn:
                res[id(s)] = up
            elif dn and not up:
                res[id(s)] = -dn
            elif up and dn:
                res[id(s)] = up if up >= dn else -dn
            else:
                res[id(s)] = 0
    return res


def render(page, base_pairs, key_pc, med_h, mode):
    P = [(s, MO.XO.OcrNote(digit=o.digit, octave=int(o.octave), acc=o.acc, aug_dot=o.aug_dot))
         for s, o in base_pairs]
    if mode == "v3":
        ov = octaves_v3(page, P, med_h)
        for s, o in P:
            o.octave = ov.get(id(s), 0)
    bgr, st = EF.render_fixed_image(page, P, key_pc, key_pc, mark_missed=False, erase_mode="comp")
    up = sum(1 for _s, o in P if int(o.octave) > 0)
    dn = sum(1 for _s, o in P if int(o.octave) < 0)
    two = sum(1 for _s, o in P if abs(int(o.octave)) >= 2)
    three = sum(1 for _s, o in P if abs(int(o.octave)) >= 3)
    both = sum(1 for s, o in P if int(o.octave) != 0)
    return bgr, dict(n=len(P), up=up, dn=dn, two=two, three=three, dotted=both)


def build(song, win_frac=0.42):
    from .glyphnet import GlyphNet
    orig, ocr_path, key_name = SONGS[song]
    txt = MO.sanitize_ocr(open(ocr_path, encoding="utf8").read())
    net = GlyphNet()
    _lines, pairs, _st, page = MO.merge_page(orig, txt, net)
    key_pc = M.parse_key(key_name)
    med_h = float(np.median([s.box[3] - s.box[1] for s, _o in pairs])) if pairs else 20.0

    old_bgr, so = render(page, pairs, key_pc, med_h, "old")
    new_bgr, sn = render(page, pairs, key_pc, med_h, "v3")

    # 裁剪窗口：取「旧算法带点最密」的行，最能对比
    per = {}
    for s, o in pairs:
        if int(o.octave) != 0:
            per[int(s.row)] = per.get(int(s.row), 0) + 1
    rows = sorted({int(s.row) for s, _o in pairs})
    focus = max(per, key=lambda r: per[r]) if per else rows[0]
    y_focus = int(min(s.box[1] for s, _o in pairs if int(s.row) == focus))
    Hh = int(page.gray.shape[0] * win_frac)
    y0 = max(0, y_focus - int(0.15 * Hh))
    y1 = min(page.gray.shape[0], y0 + Hh)

    tiles = []
    o_img = Image.open(orig).convert("RGB")
    k = page.gray.shape[1] / float(o_img.width)
    tiles.append(("原图", "参照", o_img.crop((0, int(y0 / k), o_img.width, int(y1 / k)))
                  .resize((page.gray.shape[1], y1 - y0), Image.LANCZOS)))
    for nm, stt, bgr in [("旧算法(几何连通块)", so, old_bgr), ("新算法 v3(距离变换峰值)", sn, new_bgr)]:
        tiles.append((nm, f"上点{stt['up']} 下点{stt['dn']} 双点{stt['two']} 三点{stt['three']} 带点音{stt['dotted']}",
                      Image.fromarray(bgr[y0:y1, :, ::-1])))

    Wp = max(t[2].width for t in tiles)
    BAR = 54
    hmax = max(t[2].height for t in tiles)
    cv = Image.new("RGB", (Wp, (BAR + hmax) * len(tiles) + 10), (255, 255, 255))
    d = ImageDraw.Draw(cv)
    f1, f2 = _font(27), _font(19)
    for i, (nm, sub, im) in enumerate(tiles):
        y = i * (BAR + hmax + 10)
        d.rectangle([0, y, Wp, y + BAR], fill=(245, 245, 245))
        d.text((8, y + 3), f"{i}. {nm}", font=f1, fill=(10, 10, 10))
        d.text((8, y + 31), sub, font=f2, fill=(130, 0, 0))
        cv.paste(im, (0, y + BAR))
        d.rectangle([0, y, Wp - 1, y + BAR + im.height], outline=(185, 185, 185))
    out = os.path.join("out", "dots3", f"{song}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out, quality=93, optimize=True)
    print("%-9s 旧: 上%d 下%d 双%d | 新v3: 上%d 下%d 双%d 三%d | %dx%d" % (
        song, so["up"], so["dn"], so["two"], sn["up"], sn["dn"], sn["two"], sn["three"],
        cv.width, cv.height), flush=True)
    return out


def main():
    return [build(s) for s in SONGS]


if __name__ == "__main__":
    main()
