"""Recompose real jianpu rows (taken from a real score) into controlled layouts.

The glyphs are real, only the *arrangement* is synthetic - which lets us put a
number on the failure modes the user reports:
  V1 variable line spacing          变行距
  V2 two parallel blocks            多段并行
  V3 one wide gap                   清楚的一行识别不到
  V4 very tight spacing             相邻两行被黏成一段
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from . import core


@dataclass
class Band:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def extract_bands(gray: np.ndarray, min_h: int = 6) -> List[Band]:
    """Ink bands of a real page, with horizontal trimming to real content."""
    bw = core.binarize(gray)
    clean, _ = core.strip_rules(bw)
    bands = core.split_adaptive(core.row_profile(clean), min_ink=0.5)
    out: List[Band] = []
    for y0, y1 in bands:
        strip = clean[y0:y1]
        cols = core.col_profile(strip)
        xs = np.nonzero(cols > 0)[0]
        if len(xs) == 0:
            continue
        b = Band(y0, y1, int(xs[0]), int(xs[-1]) + 1)
        if b.h >= min_h:
            out.append(b)
    return out


def _paste(canvas: np.ndarray, full: np.ndarray, band: Band, x: int, y: int, max_w: int = 0) -> Tuple[int, int, int, int]:
    crop = full[band.y0:band.y1, band.x0:band.x1]
    if max_w and crop.shape[1] > max_w:
        s = max_w / float(crop.shape[1])
        crop = cv2.resize(crop, (max_w, max(1, int(round(crop.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    h, w = crop.shape
    H, W = canvas.shape
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return (x, y, x, y)
    canvas[y:y + h, x:x + w] = np.minimum(canvas[y:y + h, x:x + w], crop[:h, :w])
    return (x, y, x + w, y + h)


def build_variant(gray: np.ndarray, bands: List[Band], mode: str) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    H, W = 1500, 1100
    canvas = np.full((H, W), 255, dtype=np.uint8)
    gts: List[Tuple[int, int, int, int]] = []
    band_h = max(b.h for b in bands)

    if mode == "variable":
        gaps = [8, 62, 120, 14, 96, 6, 74, 30, 110, 10]
        indents = [30, 120, 60, 10, 160, 40, 100, 20, 140, 70]
        y = 40
        for i, b in enumerate(bands[:10]):
            x = indents[i % len(indents)]
            gts.append(_paste(canvas, gray, b, x, y))
            y += b.h + gaps[i % len(gaps)]
    elif mode == "twoblock":
        left = bands[0::2][:5]
        right = bands[1::2][:5]
        x_l, x_r = 30, 600
        y = 40
        for b in left:
            gts.append(_paste(canvas, gray, b, x_l, y, max_w=480))
            y += b.h + 30
        y = 62  # deliberately not aligned with the left block
        for b in right:
            gts.append(_paste(canvas, gray, b, x_r, y, max_w=480))
            y += b.h + 44
    elif mode == "biggap":
        y = 40
        for i, b in enumerate(bands[:8]):
            gap = 220 if i == 4 else 12
            gts.append(_paste(canvas, gray, b, 40, y))
            y += b.h + gap
    elif mode == "tight":
        y = 40
        for b in bands[:9]:
            gts.append(_paste(canvas, gray, b, 40, y))
            y += b.h + 1
    else:
        raise ValueError(mode)
    return canvas, [g for g in gts if g[2] > g[0] and g[3] > g[1]]


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua else 0.0
