"""按小节切分 + 按小节对齐。

为什么按小节切（用户提的思路，实测数据支持）：

- 整行对齐太脆弱。OCR 会把带点音写成两个数字（`1̇` → `1 1`），一行里多出
  一两个音，按行内第 k 个音配对就会整体挪位——八度点就跑到别的音上去了。
- 小节线是**印刷出来的强锚点**，两边都有（几何侧是竖直细长墨块、OCR 输出 `|`）。
  在锚点之间局部对齐，一处数量不符只污染这一个小节，不会传染整行。
- 小节内字号基本一致（装饰音例外），所以"用本小节的中位字高做归一化尺度"
  是成立的，这对后面训练字形分类器很关键。

实测（为了你唱下去，1721px）：前 14 个小节音符数逐个对比，只有 1 个不同；
三月雨（742px、9px 字）几何侧太弱，小节数都对不上（79 vs 77）。

数据流：
    page ─► geo_measures()  ─┐
                              ├─► align_measures() ─► pairs (slot, OcrNote)
    OCR  ─► ocr_measures()  ─┘
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import align as AL
from . import core
from . import xiaomi_ocr as XO

_BAR_RE = re.compile(r"[|｜‖¦]")


# ---------------------------------------------------------------- 几何侧

def detect_bars(page: core.Page, slots: Sequence[core.Note]) -> List[core.Comp]:
    """小节线：比音符高、很窄的竖直墨块。

    不靠 `is_rule_like`（那是按页级 scale 判的，低分辨率页会误判），
    改用**本页音符的实际中位字高**做基准。
    """
    if not slots:
        return []
    med_h = float(np.median([s.box[3] - s.box[1] for s in slots]))
    out: List[core.Comp] = []
    for c in page.comps:
        if c.area < 20:
            continue
        if c.h > 1.35 * med_h and c.w < 0.30 * med_h:
            out.append(c)
    return out


def geo_measures(page: core.Page,
                 slots: Sequence[core.Note]) -> List[List[core.Note]]:
    """把槽位流按小节线切开，返回按阅读顺序的小节列表。"""
    if not slots:
        return []
    bars = detect_bars(page, slots)
    by_row: Dict[int, List[core.Note]] = {}
    for s in slots:
        by_row.setdefault(int(s.row), []).append(s)
    out: List[List[core.Note]] = []
    for _r, ss in sorted(by_row.items()):
        ss = sorted(ss, key=lambda s: s.x)
        y0 = ss[0].box[1]
        y1 = max(s.box[3] for s in ss)
        bx = sorted(float(c.x) for c in bars if c.y < y1 and c.y + c.h > y0)
        cur = 0
        g: List[core.Note] = []
        for s in ss:
            while cur < len(bx) and bx[cur] < s.x:
                out.append(g)          # 空小节也保留：它是真实存在的小节线
                g = []
                cur += 1
            g.append(s)
        out.append(g)
    return out          # 首尾的空小节是真实的小节线两侧，保留以便和 OCR 侧对齐


# ---------------------------------------------------------------- OCR 侧

def ocr_measures(raw_lines: Sequence[str]) -> Tuple[List[List[XO.OcrNote]], List[int]]:
    """把 OCR 的谱行按 `|` 切开，返回 (小节列表, 每个小节来自第几条谱行)。"""
    out: List[List[XO.OcrNote]] = []
    src: List[int] = []
    li = 0
    for line in raw_lines:
        if not XO.parse_line(line).is_music:
            continue
        for seg in _BAR_RE.split(line):
            p = XO.parse_line(seg)
            notes = [XO.OcrNote(digit=n.digit, octave=_norm(n.octave), acc=n.acc,
                                aug_dot=n.aug_dot) for n in p.notes]
            out.append(notes)
            src.append(li)
        li += 1
    return out, src


def _norm(v) -> int:
    try:
        return max(-1, min(1, int(round(float(v or 0)))))
    except Exception:
        return 0


# ---------------------------------------------------------------- 小节对齐

_SUB_CAP = 3.0
_SUB = 6.0
_GAP = -2.0


def align_measures(geo: Sequence[Sequence[core.Note]],
                   ocr: Sequence[Sequence[XO.OcrNote]]) -> List[Optional[int]]:
    """返回每个 OCR 小节对应的几何小节下标（对不上则 None）。

    代价只看**音符数**和**小节内几何数字**，不比整页，所以一处多音只影响本节。
    """
    m, n = len(ocr), len(geo)
    if m == 0:
        return []
    if n == 0:
        return [None] * m
    gd = [[s.digit for s in g] for g in geo]
    od = [[o.digit for o in o] for o in ocr]
    NEG = -1e18
    dp = np.full((m + 1, n + 1), NEG)
    bt = np.zeros((m + 1, n + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] + _GAP
        bt[0, j] = 2
    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] + _GAP
        bt[i, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diff = abs(len(od[i - 1]) - len(gd[j - 1]))
            sc = dp[i - 1, j - 1] + _SUB - min(_SUB_CAP, diff)
            move = 0
            v = dp[i - 1, j] + _GAP
            if v > sc:
                sc, move = v, 1
            v = dp[i, j - 1] + _GAP
            if v > sc:
                sc, move = v, 2
            dp[i, j] = sc
            bt[i, j] = move
    out: List[Optional[int]] = [None] * m
    i, j = m, n
    while i > 0 or j > 0:
        mv = bt[i, j]
        if i > 0 and j > 0 and mv == 0:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and mv == 1:
            i -= 1
        elif j > 0 and mv == 2:
            j -= 1
        else:
            i, j = i - 1, j - 1
    return out


def pair_by_measures(page: core.Page, slots: Sequence[core.Note],
                     raw_lines: Sequence[str]
                     ) -> Tuple[List[Tuple[core.Note, XO.OcrNote]], dict,
                                List[List[XO.OcrNote]], List[int]]:
    """小节对齐 + 小节内配对，返回 (pairs, 统计, OCR小节, 每小节来源行号)。

    小节内音符数相等就直接按序配对（最常见、也最可靠）；
    不等就在**这个小节内部**做一次单调对齐，把多出来的音局部消化掉，
    不会影响别的小节。
    """
    geo = geo_measures(page, slots)
    ocr, src = ocr_measures(raw_lines)
    mapping = align_measures(geo, ocr)
    pairs: List[Tuple[core.Note, XO.OcrNote]] = []
    st = dict(measures_geo=len(geo), measures_ocr=len(ocr),
              matched=0, exact=0, local_dp=0, unmatched=0)
    for oi, gi in enumerate(mapping):
        if gi is None:
            st["unmatched"] += len(ocr[oi])
            continue
        st["matched"] += 1
        g = geo[gi]
        o = ocr[oi]
        if len(g) == len(o):
            st["exact"] += 1
            pairs.extend(zip(g, o))
            continue
        st["local_dp"] += 1
        mp = AL.align_streams([x.digit for x in o], [s.digit for s in g])
        for k, j in enumerate(mp):
            if j is not None:
                pairs.append((g[j], o[k]))
    return pairs, st, ocr, src


def score_pairs(pairs: Sequence[Tuple[core.Note, XO.OcrNote]],
                n_ocr: int) -> float:
    """策略打分 = 数字一致率 × 覆盖率。

    只看一致率会偏向"少配一点但都配对"的策略，那样会丢掉大量音符；
    只看覆盖率会偏向乱配。两者相乘才平衡。
    """
    if not pairs or n_ocr <= 0:
        return -1.0
    agree = sum(1 for g, o in pairs if g.digit == o.digit) / len(pairs)
    cover = len(pairs) / float(n_ocr)
    return agree * min(1.0, cover)
