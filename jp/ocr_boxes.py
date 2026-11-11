"""用 RapidOCR（PaddleOCR 的纯 ONNX 实现）读简谱，**带每个字形的坐标**。

为什么换掉 Xiaomi-OCR-0（VLM）：
- VLM 只吐文本、不吐坐标，所以必须把「OCR 的数字流」和「几何检测的位置槽位」
  做跨模态对齐。整条链路上最难缠的 bug 都出自这一步：OCR 把 `1̇` 写成 `1 1`
  导致行内错位、八度点被搬到别的音上、`2̇· 3̇` 串位……全是"没有坐标"派生出来的。
- RapidOCR 的检测器直接输出**每个字形的框**，数字和位置天然绑定，
  对齐从"全局序列匹配"退化成"最近邻配对"，误差被框边界限制住。

实测（为了你唱下去，1721px）：
    '5' x=165..193 | '3' x=229..255 | '21' x=273..341 | '1' x=348..370 ...
多数框就是单个数字，偶尔相邻两个粘连（`21`、`5-0`），按宽度等分即可。
整页 28s（CPU），比 VLM 的 40~60s 还快。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core

DIGITS = set("01234567")


@dataclass
class GlyphBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class CharPos:
    """一个**字符**（不是框）的位置。多字符框按宽度等分拆开。"""
    ch: str
    cx: float
    cy: float
    w: float
    h: float


_ENGINE = None


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def read_boxes(img: np.ndarray, use_cache: bool = False, cache_path: str = ""
               ) -> Tuple[List[GlyphBox], float]:
    """对一张图（BGR 或灰度 ndarray）跑检测+识别，返回框列表与耗时。"""
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            d = json.load(open(cache_path, encoding="utf8"))
            return [GlyphBox(*b) for b in d["boxes"]], float(d.get("secs", 0.0))
        except Exception:
            pass
    t0 = time.time()
    res, _elapse = _engine()(img)
    secs = time.time() - t0
    out: List[GlyphBox] = []
    for item in (res or []):
        box, txt, sc = item[0], item[1], item[2]
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        out.append(GlyphBox(min(xs), min(ys), max(xs), max(ys), str(txt), float(sc)))
    if use_cache and cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump({"boxes": [[b.x0, b.y0, b.x1, b.y1, b.text, b.score] for b in out],
                   "secs": secs},
                  open(cache_path, "w", encoding="utf8"), ensure_ascii=False)
    return out, secs


def chars_from_boxes(boxes: Sequence[GlyphBox], digits_only: bool = True) -> List[CharPos]:
    """把框拆成逐字符位置。

    `digits_only=False` 时保留全部字符（含中文歌词）——歌词挂到音符上需要
    文字本身和它的坐标，只留数字就什么也挂不上。
    多字符框按字符数**等分宽度**：简谱是等宽字体排版，PaddleOCR 的框边界
    基本贴着首尾字符，等分误差通常在半个字宽以内，对最近邻配对够用。
    """
    out: List[CharPos] = []
    for b in boxes:
        raw = b.text
        keep = [c for c in raw if (c in DIGITS) or (not digits_only and not c.isspace())]
        if not keep:
            continue
        w = b.x1 - b.x0
        h = b.y1 - b.y0
        n = max(1, len(raw))
        cw = w / float(n)
        for i, c in enumerate(raw):
            if digits_only and c not in DIGITS:
                continue
            if (not digits_only) and c.isspace():
                continue
            out.append(CharPos(ch=c, cx=b.x0 + cw * (i + 0.5), cy=b.cy, w=cw, h=h))
    out.sort(key=lambda p: (p.cy, p.cx))
    return out


def pair_positions(page: core.Page, slots: Sequence[core.Note],
                   chars: Sequence[CharPos], max_dist_em: float = 1.1
                   ) -> List[Tuple[core.Note, str]]:
    """按位置做一对一指派：每个槽位找最近的字符，距离超过 `max_dist_em` 就不配。

    这是"有坐标"带来的最大好处——不再需要整页单调对齐。
    一处 OCR 多吐/少吐字符只会留下一个未配对的槽位，不会把后面全推错位。
    """
    if not slots or not chars:
        return []
    H = float(page.scale)
    used = [False] * len(chars)
    cand: List[Tuple[float, int, int]] = []
    for si, s in enumerate(slots):
        x0, y0, x1, y1 = s.box
        for ci, c in enumerate(chars):
            dx = abs(c.cx - s.x)
            dy = abs(c.cy - (y0 + y1) / 2.0)
            if dx > max_dist_em * H * 0.75:
                continue
            if dy > max_dist_em * H * 0.75:
                continue
            cand.append((dx + dy, si, ci))
    cand.sort()
    pairs: List[Tuple[core.Note, str]] = []
    slot_done = set()
    for d, si, ci in cand:
        if si in slot_done or used[ci]:
            continue
        slot_done.add(si)
        used[ci] = True
        pairs.append((slots[si], chars[ci].ch))
    # 按阅读顺序返回
    pairs.sort(key=lambda p: (int(p[0].row), p[0].x))
    return pairs


def read_page_chars(img: np.ndarray, cache_path: str = "") -> Tuple[List[CharPos], float]:
    boxes, secs = read_boxes(img, use_cache=bool(cache_path), cache_path=cache_path)
    return chars_from_boxes(boxes), secs
