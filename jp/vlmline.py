"""VLM 谱行读取（生产版）：以「采样收敛」为验收闸门，不收敛就换窗口重试。

实测规律（三月雨）：同一片在合适缩放下 5 次采样会有 4-5 次给出**完全相同**的序列，
此时结果是对的；不收敛的片则完全不可信。所以：
  1) 每片多采样（默认 5），要求「长度众数 + 逐位多数的支持率」达标；
  2) 不达标就平移窗口 / 换缩放再试；
  3) 全部片用重叠区 token 匹配拼接。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import vlm

TILE_W = 280
SCALES = (6.0, 5.0, 7.0)
SAMPLES = 5
MIN_SUPPORT = 0.6


@dataclass
class TileRead:
    x0: int
    x1: int
    scale: float
    tokens: List[str] = field(default_factory=list)
    support: float = 0.0
    converged: bool = False

    @property
    def ok(self) -> bool:
        return self.converged and bool(self.tokens)


def _ask_tokens(img: np.ndarray, temperature: float = 0.4) -> List[str]:
    return vlm.read_note_string(img, temperature=temperature, max_tokens=16000)


def read_tile(bgr: np.ndarray, y0: int, y1: int, x0: int, x1: int,
              scale: float, samples: int = SAMPLES) -> TileRead:
    crop = bgr[max(0, y0):y1, x0:x1]
    if crop.size == 0:
        return TileRead(x0, x1, scale)
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    seqs = []
    for _ in range(samples):
        try:
            seqs.append(_ask_tokens(big))
        except Exception:
            seqs.append([])
    seqs = [s for s in seqs if s]
    if not seqs:
        return TileRead(x0, x1, scale)
    lens = Counter(len(s) for s in seqs)
    n_mode, cnt = lens.most_common(1)[0]
    same = [s for s in seqs if len(s) == n_mode]
    support = cnt / len(seqs)
    toks = [Counter(s[i] for s in same).most_common(1)[0][0] for i in range(n_mode)]
    # 逐位支持率（取最低位，衡量最弱环节）
    per_pos = min(Counter(s[i] for s in same).most_common(1)[0][1] / len(same) for i in range(n_mode))
    converged = support >= MIN_SUPPORT and per_pos >= 0.99
    return TileRead(x0, x1, scale, toks, support, converged)


@dataclass
class LineRead:
    tokens: List[str]
    tiles: List[TileRead]
    coverage: float          # 收敛覆盖的横向比例
    ok: bool


def read_line(bgr: np.ndarray, y0: int, y1: int, W: Optional[int] = None,
              tile_w: int = TILE_W, step_ratio: float = 0.5, dbg: bool = False,
              scales: Optional[Sequence[float]] = None,
              samples: int = SAMPLES) -> LineRead:
    H, W = bgr.shape[:2] if W is None else (bgr.shape[0], W)
    step = max(60, int(tile_w * step_ratio))
    starts = list(range(0, max(1, W - tile_w + 1), step))
    if not starts or starts[-1] + tile_w < W:
        starts.append(max(0, W - tile_w))

    accepted: List[TileRead] = []
    all_tiles: List[TileRead] = []
    for x0 in starts:
        x1 = min(W, x0 + tile_w)
        best: Optional[TileRead] = None
        for sc in (scales or SCALES):
            tr = read_tile(bgr, y0, y1, x0, x1, sc, samples=samples)
            all_tiles.append(tr)
            if tr.ok and (best is None or tr.support > best.support):
                best = tr
            if tr.ok and tr.support >= 0.8:
                break
        if best is not None:
            accepted.append(best)
        if dbg:
            got = best.tokens if best else []
            print(f"    x={x0:3d}..{x1:3d} 接受={bool(best)} 支持={best.support if best else 0:.1f} "
                  f"{' '.join(got[:12])}", flush=True)

    tokens = merge_tiles(accepted)
    covered = sum(t.x1 - t.x0 for t in accepted) / max(1, W)
    return LineRead(tokens, all_tiles, min(1.0, covered), bool(accepted))


def merge_tiles(tiles: Sequence[TileRead]) -> List[str]:
    """按 x 排序，用重叠区的后缀/前缀匹配拼接。"""
    tiles = sorted([t for t in tiles if t.tokens], key=lambda t: t.x0)
    if not tiles:
        return []
    out = list(tiles[0].tokens)
    prev_x1 = tiles[0].x1
    for t in tiles[1:]:
        overlap = max(0, prev_x1 - t.x0)
        frac = overlap / max(1.0, (t.x1 - t.x0))
        k = max(1, int(round(len(t.tokens) * min(0.6, max(0.15, frac)))))
        best_shift, best_len = 0, -1
        for shift in range(1, min(k, len(t.tokens), len(out)) + 1):
            if out[-shift:] == t.tokens[:shift]:
                if shift > best_len:
                    best_shift, best_len = shift, shift
        out.extend(t.tokens[best_shift:])
        prev_x1 = t.x1
    return out
