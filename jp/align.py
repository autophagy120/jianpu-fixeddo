"""把 OCR 的数字序列和几何检测的"位置槽位"序列做全局单调对齐。

为什么不能按行索引合并（旧做法，已废弃）：

- Xiaomi-OCR 输出的是**文本行**，视觉行边界不可靠。实测第 6 行文本把两条
  视觉谱行合并成了一行（`（6i23 6i23 6i23 6i23 | 6i23 6i23 3）6i | 2·3 21 65 | ...`），
  行数对不上，任何"第 i 行对第 i 行"的做法都必然错位；
- 几何侧的分类器会把 `6` 判成"非数字"而丢弃（`build_notes` 默认行为），
  于是行内音符数也少几个，按行内第 k 个音对齐同样错位。
  实测就是这样把 `6 1̇ 2̇ 3̇` 的三个高八度点整体前挪一位，变成 `6̇ 1̇ 2 3`。

所以这里两边都当成**一条按阅读顺序排好的序列**，在整页范围内做
Needleman-Wunsch 单调对齐（几何槽位 = 所有占位的字形，不管认没认出来）。
单调性是硬约束：读谱顺序不会倒回去，因此对齐只会插入/跳过，不会乱序。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import core


# ---------------------------------------------------------------- 槽位提取

def ordered_slots(page: core.Page, net=None) -> List[core.Note]:
    """整页的"位置槽位"，按阅读顺序排好，每个槽位带 `row` = 视觉行号。

    槽位 = 谱行内所有字形大小的连通块，**包括分类器认不出的**（digit='?'）。
    行号不取几何行带：低分辨率页会把页眉/谱行/歌词粘成一条带
    （实测三月雨第 0 条音乐行带 y 跨 316..453，等于 4.7 个字高），
    而是在行内再按字形顶部的 y 聚类，簇即视觉行。
    """
    H = float(page.scale)
    slots: List[core.Note] = []
    line_no = 0
    for r in page.rows:
        if r.kind != "music":
            continue
        got = core.build_notes(page.clean, r, net, page.scale, row_index=0,
                               dash=page.dash, preset=None, keep_unknown=True)
        if not got:
            continue
        # 行内按 y 分簇：相邻字形顶部差 > 0.6 字高就算另一条视觉行
        tops = np.array(sorted({round(float(n.box[1]), 1) for n in got}), dtype=np.float64)
        bounds = [-1e9]
        if len(tops) > 1:
            bounds += [float(tops[k + 1]) for k in np.nonzero(np.diff(tops) > 0.6 * H)[0]]
        bounds.append(1e9)

        def cluster(y: float) -> int:
            for i in range(len(bounds) - 1):
                if bounds[i] <= y < bounds[i + 1]:
                    return i
            return len(bounds) - 2

        for n in got:
            n.row = cluster(float(n.box[1]))
        got.sort(key=lambda n: (n.row, n.x))
        cur = None
        for n in got:
            if n.row != cur:          # 簇换了就是新的一行
                cur = n.row
                line_no += 1
            n.row = line_no - 1
            slots.append(n)
    return slots


# ---------------------------------------------------------------- 对齐打分

_MATCH = 3.0        # 两边都认出且一致
_UNKNOWN = 1.2      # 几何侧认不出（'?'），但位置对得上 —— 仍然可信
_MISMATCH = -0.2    # 两边都认出但不一致（几何分类器本来就常错，轻罚）
_GAP_OCR = -1.8     # OCR 有此音、几何没槽位（几何丢块）
_GAP_GEO = -0.7     # 几何多此槽位（点被当成字形、括号等）


def _score(d_ocr: str, d_geo: Optional[str]) -> float:
    if d_geo is None or d_geo == "?":
        return _UNKNOWN
    if d_ocr == d_geo:
        return _MATCH
    return _MISMATCH


def align_streams(ocr_digits: Sequence[str],
                  geo_digits: Sequence[Optional[str]]) -> List[Optional[int]]:
    """返回每个 OCR 音符对应的几何槽位下标（无对应则 None）。"""
    m, n = len(ocr_digits), len(geo_digits)
    if m == 0:
        return []
    if n == 0:
        return [None] * m
    NEG = -1e18
    # dp[i][j] = 前 i 个 OCR 音符与前 j 个槽位对齐的最大分
    dp = np.full((m + 1, n + 1), NEG, dtype=np.float64)
    bt = np.zeros((m + 1, n + 1), dtype=np.int8)   # 0=diag 1=skip ocr 2=skip geo
    dp[0, 0] = 0.0
    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] + _GAP_GEO
        bt[0, j] = 2
    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] + _GAP_OCR
        bt[i, 0] = 1
    for i in range(1, m + 1):
        di = ocr_digits[i - 1]
        for j in range(1, n + 1):
            best = dp[i - 1, j - 1] + _score(di, geo_digits[j - 1])
            move = 0
            v = dp[i - 1, j] + _GAP_OCR
            if v > best:
                best, move = v, 1
            v = dp[i, j - 1] + _GAP_GEO
            if v > best:
                best, move = v, 2
            dp[i, j] = best
            bt[i, j] = move
    # 回溯
    mapping: List[Optional[int]] = [None] * m
    i, j = m, n
    while i > 0 or j > 0:
        mv = bt[i, j]
        if i > 0 and j > 0 and mv == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and mv == 1:
            mapping[i - 1] = None
            i -= 1
        elif j > 0 and mv == 2:
            j -= 1
        else:                                   # 边界兜底
            i, j = i - 1, j - 1
    return mapping


def apply_octaves(ocr_notes: Sequence, geo_by_ocr: Sequence[Optional[core.Note]]) -> int:
    """把几何八度搬到 OCR 音符上，返回被改动的个数。

    规则（几何优先）：
    - 几何检测到点 -> 用几何的（点是小墨块，几何按位置判定很稳，OCR 会丢）；
    - 几何没点、OCR 写了点（`i`）-> 保留 OCR 的（互相印证，不冲突）；
    - 两边都有但方向相反 -> 信几何。
    """
    changed = 0
    for o, g in zip(ocr_notes, geo_by_ocr):
        if g is None:
            continue
        go = max(-2, min(2, int(g.octave)))
        oo = max(-2, min(2, int(getattr(o, "octave", 0) or 0)))
        new = go if go != 0 else oo
        if new != oo:
            o.octave = new
            changed += 1
    return changed
