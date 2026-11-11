"""自举标注：用 VLM 已确认的音符当模板，把同行剩下没确认的认出来。

同一页同字体同字号，模板与待测字形几乎像素级重合（实测簇内 IoU 0.94-1.00），
所以在**原生尺寸**上做 1-NN，热启动标签来自 VLM 的高置信结果。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core
from . import templbank as TB


def _mask_of(page: core.Page, n: core.Note) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = [int(v) for v in n.box]
    h, w = page.clean.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = page.clean[y0:y1, x0:x1]
    return TB.norm_mask(TB.clean(patch))


def propagate(page: core.Page, min_iou: float = 0.84, margin: float = 0.04,
              dbg: bool = False) -> Dict[str, int]:
    """就地给未确认音符打标（标 propagated=True）。"""
    stat = {"templates": 0, "filled": 0, "left": 0}
    for sysn in page.systems:
        notes = [n for n in sysn.notes if n.digit != "0"]
        seeds = [n for n in notes if getattr(n, "confirmed", False)]
        todo = [n for n in notes if not getattr(n, "confirmed", False)]
        if len(seeds) < 3 or not todo:
            stat["left"] += len(todo)
            continue
        tpl: List[Tuple[str, np.ndarray]] = []
        for n in seeds:
            m = _mask_of(page, n)
            if m is not None:
                tpl.append((n.digit, m))
        stat["templates"] += len(tpl)
        if not tpl:
            stat["left"] += len(todo)
            continue
        for n in todo:
            m = _mask_of(page, n)
            if m is None:
                stat["left"] += 1
                continue
            scored = []
            for d, t in tpl:
                u = np.logical_or(m > 0, t > 0).sum()
                s = float(np.logical_and(m > 0, t > 0).sum()) / u if u else 0.0
                scored.append((s, d))
            scored.sort(reverse=True)
            best_s, best_d = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if best_s >= min_iou and (best_s - second) >= margin:
                n.digit = best_d
                n.conf = 0.8
                n.propagated = True
                stat["filled"] += 1
            else:
                stat["left"] += 1
        if dbg:
            print(f"    行{sysn.row_index}: 种子 {len(seeds)} -> 补 {stat['filled']} / 剩 {stat['left']}",
                  flush=True)
    return stat
