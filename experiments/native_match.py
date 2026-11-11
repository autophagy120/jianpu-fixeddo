"""原尺寸模板匹配：小字（<12 px）不要缩放归一化，直接在原始像素上比。

这是 easytr 的做法（templates.npy = 1009 x 30 维特征，reco_note 在 position_region
里取原尺寸小块做最近邻）。把 6x9 px 的 `6` 放大到 32x32 再交给 CNN，笔画会被糊成一团；
在原始尺寸上比对反而分得开。
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DIGITS = list("01234567")

FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "georgia.ttf", "calibri.ttf",
    "cambria.ttc", "verdana.ttf", "tahoma.ttf", "simsun.ttc", "simhei.ttf", "msyh.ttc",
    "Deng.ttf", "BOOKOS.TTF", "consola.ttf", "STSONG.TTF", "GARA.TTF", "CENTURY.TTF",
]


def _fonts() -> List[str]:
    base = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    return [os.path.join(base, f) for f in FONT_CANDIDATES if os.path.exists(os.path.join(base, f))]


def _render(ch: str, font_path: str, size: int) -> Optional[np.ndarray]:
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size, size), ch, fill=255, font=font)
    a = np.array(img)
    ink = (a > 120).astype(np.uint8) * 255
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return None
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_bank(heights: Sequence[int], widths: Optional[Sequence[int]] = None) -> List[Tuple[str, np.ndarray]]:
    """按"目标字高"渲染一批原尺寸模板（不高倍渲染再缩小）。"""
    bank: List[Tuple[str, np.ndarray]] = []
    fonts = _fonts()
    for h in sorted(set(int(x) for x in heights if x >= 4)):
        for fp in fonts:
            for fs in range(max(6, h - 4), h + 12):
                for ch in DIGITS:
                    m = _render(ch, fp, fs)
                    if m is None:
                        continue
                    if abs(m.shape[0] - h) <= 1:
                        bank.append((ch, m))
    return bank


def match(patch: np.ndarray, bank: Sequence[Tuple[str, np.ndarray]], min_iou: float = 0.55
          ) -> Tuple[Optional[str], float, float]:
    """返回 (标签, 最佳 IoU, 一二名间距)。patch 为 ink=255 的二值小块（原尺寸）。"""
    if patch.size == 0 or not bank:
        return None, 0.0, 0.0
    a = (patch > 0)
    best = ("", -1.0)
    second = -1.0
    for ch, t in bank:
        if abs(t.shape[0] - a.shape[0]) > 2 or abs(t.shape[1] - a.shape[1]) > 3:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                tt = t
                aa = a
                if dy or dx:
                    aa = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
                h = min(aa.shape[0], tt.shape[0])
                w = min(aa.shape[1], tt.shape[1])
                if h <= 0 or w <= 0:
                    continue
                aa2, tt2 = aa[:h, :w], tt[:h, :w] > 0
                inter = np.logical_and(aa2, tt2).sum()
                union = np.logical_or(aa2, tt2).sum()
                if union == 0:
                    continue
                iou = inter / union
                if iou > best[1]:
                    second = best[1]
                    best = (ch, iou)
                elif iou > second:
                    second = iou
    if best[1] < 0:
        return None, 0.0, 0.0
    return best[0], float(best[1]), float(best[1] - max(0.0, second))
