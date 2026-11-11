"""用「语料级模板库」替代 CNN 认数字（原生分辨率 1-NN）。

与 GlyphNet 的分工：
- 数字：本模块（原生尺寸 + 模板库，字体匹配时几乎像素级重合）
- 升降号/八度点/行类型：仍由 CNN 与几何负责（它们的字形与数字差别大，CNN 够用）
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import templbank as TB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BANK_PATH = os.path.join(ROOT, "models", "templ_bank.npz")


class TemplBank:
    def __init__(self, path: str = BANK_PATH, min_sim: float = 0.62, margin: float = 0.02):
        z = np.load(path, allow_pickle=True)
        cs = z["centroids"]
        labels = [str(x) for x in z["labels"]]
        self.min_sim = min_sim
        self.margin = margin
        keep = [i for i, l in enumerate(labels) if l in "01234567"]
        if not keep:
            raise RuntimeError("模板库还没有命名的簇，先跑 python -m jp.banklabel")
        self.centroids = cs[keep]
        self.labels = [labels[i] for i in keep]
        self.counts = z["counts"][keep]
        print(f"[TemplBank] {len(keep)} 个已命名模板，覆盖 {int(self.counts.sum())} 个字形实例")

    # ---- 与 GlyphNet 对齐的接口 ----
    def digit(self, patches: Sequence[np.ndarray]) -> Tuple[List[Optional[str]], np.ndarray]:
        out: List[Optional[str]] = []
        confs = np.zeros(len(patches), dtype=np.float32)
        if not patches:
            return out, confs
        masks = [TB.norm_mask(TB.clean(p)) for p in patches]
        for i, m in enumerate(masks):
            scored = []
            for lab, c in zip(self.labels, self.centroids):
                u = np.logical_or(m > 0, c > 0).sum()
                s = float(np.logical_and(m > 0, c > 0).sum()) / u if u else 0.0
                scored.append((s, lab))
            scored.sort(reverse=True)
            best_s, best_l = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            confs[i] = best_s
            if best_s < self.min_sim or (best_s - second) < self.margin:
                out.append(None)
            else:
                out.append(best_l)
        return out, confs


def make(path: str = BANK_PATH, **kw) -> Optional[TemplBank]:
    try:
        return TemplBank(path, **kw)
    except Exception as exc:
        print(f"[TemplBank] 不可用: {exc}")
        return None
