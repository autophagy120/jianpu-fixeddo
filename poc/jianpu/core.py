"""POC core for jianpu (numbered notation) page analysis.

Three things live here, deliberately separated so they can be benchmarked
independently:

1. image ops      -> binarise, strip the long rule lines, connected components
2. layout         -> row bands (music / lyric / header) and page columns
3. glyph          -> shape features + nearest-template classifier **with a
                     reject path** (the 2018 program has no reject path)

The "baseline" functions are a faithful-in-spirit reconstruction of the
heuristics found in the shipped binaries (global median/std of section
heights, a relative ink threshold, 30-dim feature + 1009 prototypes 1-NN).
They are not the original source - the package ships only Cython .pyd files.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

INK = 255


# --------------------------------------------------------------------------
# 1. image ops
# --------------------------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    # cv2.imread cannot handle non-ASCII paths on Windows -> decode from bytes
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_bgr(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(path)
    buf.tofile(path)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold, returned with ink = 255."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def strip_rules(bw: np.ndarray, h_len: int = 22, v_len: int = 22) -> Tuple[np.ndarray, np.ndarray]:
    """Remove long horizontal/vertical rules (barlines, underlines, staff-ish lines).

    Returns (clean, rules) where `rules` keeps the removed pixels.
    Jianpu uses underlines to group notes; they glue several digits into one
    connected component, which is one reason naive CC segmentation fails.
    """
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    rules = cv2.bitwise_or(hor, ver)
    clean = cv2.bitwise_and(bw, cv2.bitwise_not(rules))
    return clean, rules


@dataclass
class Comp:
    x: int
    y: int
    w: int
    h: int
    area: int
    mask: np.ndarray = field(repr=False, default=None)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def components(bw: np.ndarray, min_area: int = 6) -> List[Comp]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out: List[Comp] = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        out.append(Comp(x, y, w, h, a))
    out.sort(key=lambda c: (c.y, c.x))
    return out


# --------------------------------------------------------------------------
# 2. layout
# --------------------------------------------------------------------------
def row_profile(bw: np.ndarray) -> np.ndarray:
    return bw.astype(np.float64).sum(axis=1) / INK


def col_profile(bw: np.ndarray) -> np.ndarray:
    return bw.astype(np.float64).sum(axis=0) / INK


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def split_baseline(prof: np.ndarray, rel_thr: float = 0.06, k: float = 1.0) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """2018-style row split: relative ink threshold + global median/std filter.

    Any band whose height falls outside [median - k*std, median + k*std] is
    *dropped* - which is exactly how a perfectly clear line can silently
    disappear from the result.
    """
    thr = max(1.0, rel_thr * prof.max())
    bands = _runs(prof > thr)
    if not bands:
        return [], []
    heights = np.array([b[1] - b[0] for b in bands], dtype=np.float64)
    med, std = float(np.median(heights)), float(heights.std())
    keep, drop = [], []
    for b in bands:
        h = b[1] - b[0]
        (keep if med - k * std <= h <= med + k * std else drop).append(b)
    return keep, drop


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(x, ker, mode="same")


def estimate_line_pitch(prof: np.ndarray) -> float:
    """Dominant line pitch from the autocorrelation of the row profile."""
    x = prof - prof.mean()
    if not np.any(x):
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo = 12
    hi = max(lo + 4, min(len(ac), 240))
    if hi <= lo:
        return 0.0
    seg = ac[lo:hi]
    return float(lo + int(np.argmax(seg)))


def split_adaptive(prof: np.ndarray, min_ink: float = 0.5) -> List[Tuple[int, int]]:
    """Spacing-agnostic row split that never throws a band away.

    1. every run of *any* ink becomes a seed band
    2. seeds closer than 45% of the estimated local line height are merged
       (a music line and its lyric line stay separate; stray accents do not)
    3. bands that are only ~an accent high are attached to their neighbour
    4. remaining bands are returned in full - the caller labels them
    """
    p = _smooth(prof, 3)
    seeds = _runs(p > min_ink)
    if not seeds:
        return []
    heights = np.array([b[1] - b[0] for b in seeds], dtype=np.float64)
    unit = float(np.median(heights))  # height of the most common band kind
    pitch = estimate_line_pitch(prof) or 2.2 * unit
    gap_limit = max(2.0, 0.45 * unit * 1.6)

    merged: List[Tuple[int, int]] = [seeds[0]]
    for y0, y1 in seeds[1:]:
        py0, py1 = merged[-1]
        gap = y0 - py1
        tiny_prev = (py1 - py0) < 0.55 * unit
        tiny_cur = (y1 - y0) < 0.55 * unit
        if gap <= gap_limit and (tiny_prev or tiny_cur):
            merged[-1] = (py0, max(py1, y1))
        elif gap <= 0:
            merged[-1] = (py0, max(py1, y1))
        else:
            merged.append((y0, y1))

    # attach stray accent-only bands to the nearest neighbour
    out: List[Tuple[int, int]] = []
    for band in merged:
        if not out:
            out.append(band)
            continue
        h = band[1] - band[0]
        gap = band[0] - out[-1][1]
        if h < 0.45 * unit and gap <= 4:
            py0, py1 = out[-1]
            out[-1] = (min(py0, band[0]), max(py1, band[1]))
        else:
            out.append(band)
    return out


def find_columns(bw: np.ndarray, min_gutter_frac: float = 0.035) -> List[Tuple[int, int]]:
    """Split the page into side-by-side blocks when a wide white gutter exists.

    The 2018 program has no notion of columns at all, so two parallel blocks
    (two lyric verses, a coda set beside the body, ...) get merged row-wise.
    """
    h, w = bw.shape
    prof = _smooth(col_profile(bw), 3)
    min_gutter = max(12, int(min_gutter_frac * w))
    empty = prof <= 0.3
    gutters = [r for r in _runs(empty) if r[1] - r[0] >= min_gutter]
    if not gutters:
        return [(0, w)]
    total = prof.sum() + 1e-9
    cols: List[Tuple[int, int]] = []
    start = 0
    for g0, g1 in gutters:
        left_ink = prof[:g0].sum()
        right_ink = prof[g1:].sum()
        if left_ink < 0.12 * total or right_ink < 0.12 * total:
            continue
        cols.append((start, g0))
        start = g1
    cols.append((start, w))
    cols = [c for c in cols if c[1] - c[0] > 0.05 * w]
    return cols or [(0, w)]


# --------------------------------------------------------------------------
# 3. glyph shapes
# --------------------------------------------------------------------------
NORM = 16


def norm_patch(binary_patch: np.ndarray) -> np.ndarray:
    """Binary glyph -> 16x16 float mask in [0,1], aspect ratio preserved (padded)."""
    ys, xs = np.nonzero(binary_patch)
    if len(ys) == 0:
        return np.zeros((NORM, NORM), dtype=np.float32)
    sub = binary_patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    s = NORM / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    res = cv2.resize(sub.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((NORM, NORM), dtype=np.float32)
    oy, ox = (NORM - nh) // 2, (NORM - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = (res > 0).astype(np.float32)
    return canvas


def hole_count(binary_patch: np.ndarray) -> int:
    """Number of enclosed background regions (Euler number complement)."""
    p = (binary_patch > 0).astype(np.uint8)
    if p.sum() == 0:
        return 0
    pad = np.pad(p, 1, mode="constant", constant_values=0)
    bg = (pad == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(bg, connectivity=4)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    holes = 0
    for lab in range(1, n):
        if lab in border_labels:
            continue
        if (labels == lab).sum() >= 3:
            holes += 1
    return holes


@dataclass
class GlyphFeatures:
    patch: np.ndarray
    holes: int
    aspect: float      # w/h
    fill: float        # ink / bbox
    height: int


def glyph_features(bw: np.ndarray, comp: Comp) -> GlyphFeatures:
    sub = bw[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w]
    sub = ((sub > 0).astype(np.uint8) * 255)
    return GlyphFeatures(
        patch=norm_patch(sub),
        holes=hole_count(sub),
        aspect=comp.w / max(1.0, float(comp.h)),
        fill=comp.area / max(1.0, float(comp.w * comp.h)),
        height=comp.h,
    )


# ---- templates rendered with a real font -------------------------------
SYMBOLS = ["0", "1", "2", "3", "4", "5", "6", "7"]


def render_glyph(ch: str, size: int = 64, font_path: str = r"C:\Windows\Fonts\arial.ttf") -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (size * 2, size * 2), 0)
    d = ImageDraw.Draw(img)
    d.text((size // 2, size // 2), ch, fill=255, font=font)
    a = np.array(img)
    ys, xs = np.nonzero(a > 128)
    if len(ys) == 0:
        return np.zeros((NORM, NORM), dtype=np.float32)
    return norm_patch(((a > 128).astype(np.uint8) * 255)[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


class NearestTemplate:
    """k=1 nearest neighbour over rendered digit masks (multi-font templates).

    `reject=False` reproduces the 2018 behaviour: always answer some digit.
    `reject=True` adds calibration on top of the same scores.
    """

    FONTS = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\simhei.ttf"]

    def __init__(self, symbols: Sequence[str] = SYMBOLS, font_paths: Optional[Sequence[str]] = None):
        self.symbols = list(symbols)
        paths = list(font_paths or self.FONTS)
        tpl = []
        for fp in paths:
            try:
                tpl.append(np.stack([render_glyph(s, font_path=fp) for s in self.symbols]))
            except OSError:
                continue
        self.templates = np.concatenate(tpl, axis=0)          # (n_fonts*n_symbols, NORM, NORM)
        self.tpl_symbol = [s for _ in paths for s in self.symbols]

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a > 0.5, b > 0.5).sum()
        union = np.logical_or(a > 0.5, b > 0.5).sum()
        return inter / union if union else 0.0

    def scores(self, feats: GlyphFeatures) -> np.ndarray:
        """Best IoU per symbol, in `self.symbols` order."""
        raw = np.array([self._iou(feats.patch, t) for t in self.templates])
        best = np.zeros(len(self.symbols), dtype=np.float64)
        for i, s in enumerate(self.tpl_symbol):
            j = self.symbols.index(s)
            best[j] = max(best[j], raw[i])
        return best

    def predict(
        self,
        feats: GlyphFeatures,
        reject: bool = False,
        min_iou: float = 0.55,
        min_margin: float = 0.06,
        max_holes: int = 2,
        aspect_max: float = 0.95,
    ) -> Tuple[Optional[str], float]:
        sc = self.scores(feats)
        order = np.argsort(-sc)
        best, second = sc[order[0]], sc[order[1]] if len(sc) > 1 else 0.0
        label = self.symbols[order[0]]
        if not reject:
            return label, float(best)
        if best < min_iou:
            return None, float(best)
        if best - second < min_margin:
            return None, float(best)
        if feats.holes > max_holes:
            return None, float(best)
        if feats.aspect > aspect_max:
            return None, float(best)
        return label, float(best)
