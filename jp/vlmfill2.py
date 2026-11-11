"""把 VLM 收敛读取的 token 落到音符上：按比例对齐，允许插入漏检、允许留空待核对。

原则（对应"我漏检了怎么办"）：
- 只用**已收敛**的片；
- 片内 token 数与框数相等 -> 一一对应（最可信）；
- 不等 -> 用「token 在片内的相对位置」与框的实际 x 做**单调对齐**：
    * 有框匹配到的 token -> 写回该框（已确认）
    * 没有框可配的 token -> **插入**一个新音符（已确认，x 由相对位置推出）
    * 没被任何 token 覆盖的框 -> 保持原判（待核对）
  这样漏检会被补上，而不是让整行错位。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core
from . import vlmline

TOKEN_RE = re.compile(r"^([#b]?)([0-7])(\^{0,2}|_{0,2})$")


def parse_token(tok: str) -> Optional[Tuple[str, str, int]]:
    m = TOKEN_RE.match(tok)
    if not m:
        return None
    acc, digit, marks = m.groups()
    octv = marks.count("^") if "^" in marks else -marks.count("_")
    return acc, digit, octv


@dataclass
class TileRecord:
    x0: int
    x1: int
    tokens: List[str] = field(default_factory=list)
    support: float = 0.0

    def as_list(self):
        return [self.x0, self.x1, self.tokens, self.support]


@dataclass
class FillResult:
    confirmed: int = 0
    inserted: int = 0
    uncertain: int = 0
    tiles_used: int = 0
    tiles_dropped: int = 0
    coverage: float = 0.0
    tokens_dropped: int = 0      # VLM 多给、但落点没有墨迹的 token（幻觉）
    tiles: List[TileRecord] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {"confirmed": self.confirmed, "inserted": self.inserted,
                "tokens_dropped": self.tokens_dropped,
                "uncertain": self.uncertain, "tiles_used": self.tiles_used,
                "tiles_dropped": self.tiles_dropped, "coverage": round(self.coverage, 3),
                "tiles": [t.as_list() for t in self.tiles]}


def _monotonic_align(tok_rel: Sequence[float], box_rel: Sequence[float],
                     max_dist: float = 0.12) -> List[Tuple[int, int]]:
    """单调最近对齐：返回 [(token_idx, box_idx)]。"""
    pairs: List[Tuple[int, int]] = []
    bi = 0
    for ti, tx in enumerate(tok_rel):
        best, bestd = -1, 1e9
        for j in range(bi, len(box_rel)):
            d = abs(box_rel[j] - tx)
            if d < bestd:
                best, bestd = j, d
            if box_rel[j] > tx + max_dist:
                break
        if best >= 0 and bestd <= max_dist:
            pairs.append((ti, best))
            bi = best + 1
    return pairs


def apply_tiles(page, system, tiles: Sequence[TileRecord], dbg: bool = False,
                components=None) -> FillResult:
    """离线应用（不调 API）：tiles 里是已经读好的片结果。"""
    res = FillResult(tiles=list(tiles))
    k = page.k or 1.0
    notes = sorted([n for n in system.notes if n.digit != "0"], key=lambda n: n.x)
    if not notes:
        return res
    for n in notes:
        n.confirmed = False
    # 可插入位置 = 该行"字形级"组件中，没有被任何音符框覆盖的那些中心
    spare = []
    if components:
        taken = [(min(n.box[0], n.box[2]), max(n.box[0], n.box[2])) for n in notes]
        boxes = [(min(c.x, c.x + c.w), max(c.x, c.x + c.w)) for c in components]
        for (c0, c1) in boxes:
            cx = (c0 + c1) / 2.0
            if any(a - 2 <= cx <= b + 2 for a, b in taken):
                continue
            spare.append(cx)
    base_y = int(np.median([n.box[1] for n in notes]))
    base_h = float(np.median([n.box[3] - n.box[1] for n in notes]) or 12)
    base_w = float(np.median([n.box[2] - n.box[0] for n in notes]) or 8)

    for tile in tiles:
        toks = tile.tokens
        if not toks:
            continue
        in_tile = [n for n in notes if tile.x0 <= n.x / k <= tile.x1]
        if not in_tile:
            continue
        res.tiles_used += 1
        span = max(1, tile.x1 - tile.x0)

        if len(toks) == len(in_tile):
            for n, t in zip(in_tile, toks):
                p = parse_token(t)
                if not p:
                    continue
                acc, d, o = p
                n.digit, n.conf, n.confirmed = d, 0.95, True
                if o:
                    n.octave = o
                if acc in "#b":
                    n.acc = acc
                res.confirmed += 1
            continue

        tok_rel = [(i + 0.5) / len(toks) for i in range(len(toks))]
        box_rel = [(n.x / k - tile.x0) / span for n in in_tile]
        pairs = _monotonic_align(tok_rel, box_rel)
        matched_tok = set()
        for ti, bi in pairs:
            p = parse_token(toks[ti])
            if not p:
                continue
            acc, d, o = p
            n = in_tile[bi]
            n.digit, n.conf, n.confirmed = d, 0.9, True
            if o:
                n.octave = o
            if acc in "#b":
                n.acc = acc
            matched_tok.add(ti)
            res.confirmed += 1
        for ti, rel in enumerate(tok_rel):
            if ti in matched_tok:
                continue
            p = parse_token(toks[ti])
            if not p:
                continue
            acc, d, o = p
            x = tile.x0 + rel * span                  # 原图 x
            xw = x * k                                # 工作坐标
            # 只在"有墨迹却没被识别"的位置插入；否则这个 token 是幻觉，丢弃
            if spare:
                near = min(spare, key=lambda sx: abs(sx - xw))
                if abs(near - xw) > max(0.8 * base_w, 0.6 * (span / max(1, len(toks)))):
                    res.tokens_dropped += 1
                    continue
                xw = near
                spare.remove(near)
            else:
                res.tokens_dropped += 1
                continue
            new = core.Note(digit=d, acc=acc if acc in "#b" else "", octave=o, x=xw,
                            box=(int(xw - base_w / 2), base_y,
                                 int(xw + base_w / 2), int(base_y + base_h)),
                            conf=0.7, row=system.row_index)
            new.confirmed = True
            notes.append(new)
            res.inserted += 1
        if dbg:
            print(f"    片 x={tile.x0}..{tile.x1}: token {len(toks)} 框 {len(in_tile)} "
                  f"→ 配 {len(pairs)} 插 {res.inserted} 丢 {res.tokens_dropped}", flush=True)

    notes.sort(key=lambda n: n.x)
    system.notes = notes
    res.uncertain = sum(1 for n in notes if not getattr(n, "confirmed", False))
    return res


def read_system(page, system, bgr, y0: int, y1: int, dbg: bool = False,
                scales=None, samples: int = 3, step_ratio: float = 0.85) -> FillResult:
    lr = vlmline.read_line(bgr, y0, y1, dbg=dbg, scales=scales, samples=samples,
                           step_ratio=step_ratio)
    tiles = [TileRecord(t.x0, t.x1, t.tokens, t.support) for t in lr.tiles if t.ok]
    res = apply_tiles(page, system, tiles, dbg=dbg)
    res.coverage = lr.coverage
    return res
