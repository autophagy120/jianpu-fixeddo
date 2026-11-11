"""Benchmark: 2018-style global row split vs column-aware adaptive split.

Rows are taken from a real score (成都, published with the project) and recomposed
into four controlled layouts, so the metric is "did this text row survive into
the output", which is exactly what the user perceives as 漏行 / 串行.

Run:  python bench_layout.py
Out:  out/layout_bench.json, out/layout_bench.png
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jianpu import core, realpage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SAMPLE = os.path.join(os.path.dirname(HERE), "work", "samples", "chengdu.jpg")

Box = Tuple[int, int, int, int]


# ---- metrics ------------------------------------------------------------
def coverage(gts: List[Box], preds: List[Box]) -> Dict[str, int]:
    """A GT row counts as detected only if ONE predicted band covers it.

    A band that swallows two rows (merge) therefore does not count as two
    detections - which is how a reader experiences the failure too.
    """
    covered = 0
    for (x0, y0, x1, y1) in gts:
        gh, gw = max(1, y1 - y0), max(1, x1 - x0)
        for (px0, py0, px1, py1) in preds:
            vy = max(0, min(y1, py1) - max(y0, py0)) / gh
            hx = max(0, min(x1, px1) - max(x0, px0)) / gw
            if vy >= 0.8 and hx >= 0.5:
                covered += 1
                break
    merged, spurious = 0, 0
    for (px0, py0, px1, py1) in preds:
        touched = 0
        for (x0, y0, x1, y1) in gts:
            vy = max(0, min(y1, py1) - max(y0, py0)) / max(1, y1 - y0)
            hx = max(0, min(x1, px1) - max(x0, px0)) / max(1, x1 - x0)
            if vy >= 0.5 and hx >= 0.3:
                touched += 1
        if touched >= 2:
            merged += 1
        elif touched == 0:
            spurious += 1
    return {"gt": len(gts), "pred": len(preds), "covered": covered,
            "missed": len(gts) - covered, "merged": merged, "spurious": spurious}


# ---- the two segmenters -------------------------------------------------
def boxes_baseline(clean: np.ndarray, rel_thr: float = 0.02, k: float = 1.0) -> Tuple[List[Box], int]:
    h, w = clean.shape
    prof = core._smooth(core.row_profile(clean), 5)
    runs = core._runs(prof > rel_thr * prof.max())
    if not runs:
        return [], 0
    heights = np.array([b[1] - b[0] for b in runs], dtype=np.float64)
    med, std = float(np.median(heights)), float(heights.std())
    keep, drop = [], 0
    for b in runs:
        hh = b[1] - b[0]
        if med - k * std <= hh <= med + k * std:
            keep.append(b)
        else:
            drop += 1
    return [(0, y0, w, y1) for y0, y1 in keep], drop


def boxes_adaptive(clean: np.ndarray) -> Tuple[List[Box], int]:
    out: List[Box] = []
    cols = core.find_columns(clean)
    for cx0, cx1 in cols:
        for y0, y1 in core.split_adaptive(core.row_profile(clean[:, cx0:cx1])):
            out.append((cx0, y0, cx1, y1))
    return out, len(cols)


# ---- plotting ----------------------------------------------------------
def draw(ax, img, gts, preds, title, m=None):
    from matplotlib.patches import Rectangle

    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for (x0, y0, x1, y1) in preds:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1f77b4", lw=1.0))
    for (x0, y0, x1, y1) in gts:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#2ca02c", lw=1.6, ls="--"))
    sub = ""
    if m:
        sub = f"\n检出 {m['covered']}/{m['gt']}  漏 {m['missed']}  粘连 {m['merged']}  虚检 {m['spurious']}"
    ax.set_title(title + sub, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    os.makedirs(OUT, exist_ok=True)
    gray = core.load_gray(SAMPLE)
    bands = realpage.extract_bands(gray)
    print(f"extracted {len(bands)} real text rows from {os.path.basename(SAMPLE)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    modes = [("variable", "V1 变行距"), ("twoblock", "V2 两段并行"), ("biggap", "V3 一处大空隙"), ("tight", "V4 极紧行距")]
    results: Dict[str, dict] = {}
    fig, axes = plt.subplots(len(modes), 3, figsize=(13.5, 3.9 * len(modes)))

    for row, (mode, label) in enumerate(modes):
        page, gts = realpage.build_variant(gray, bands, mode)
        bw = core.binarize(page)
        clean, _ = core.strip_rules(bw)

        # "as shipped": relative threshold + median/std gate at k=1
        pb0, drop0 = boxes_baseline(clean, rel_thr=0.02, k=1.0)
        m0 = coverage(gts, pb0)
        # best hand-tuning of that same recipe: give the baseline every advantage
        best = None
        for k in (1.0, 1.5, 2.0, 2.5, 3.0):
            pk, _ = boxes_baseline(clean, rel_thr=0.02, k=k)
            mk = coverage(gts, pk)
            if best is None or mk["covered"] > best[1]["covered"]:
                best = (k, mk, pk)
        k_best, mb, pb = best

        pa, ncols = boxes_adaptive(clean)
        ma = coverage(gts, pa)
        results[mode] = {
            "label": label,
            "baseline_as_shipped": {**m0, "dropped_by_height_gate": drop0},
            "baseline_best_tuned": {**mb, "k": k_best},
            "adaptive": {**ma, "columns_found": ncols},
        }
        print(f"{label}: GT={len(gts)}  基线(默认) 检出{m0['covered']} 漏{m0['missed']} 粘连{m0['merged']}   "
              f"基线(调优k={k_best}) 检出{mb['covered']} 漏{mb['missed']} 粘连{mb['merged']}   "
              f"自适应 检出{ma['covered']} 漏{ma['missed']} 粘连{ma['merged']} 虚检{ma['spurious']} 列数{ncols}")

        draw(axes[row, 0], page, gts, [], f"{label} / 真值(行带)")
        draw(axes[row, 1], page, gts, pb, "2018式:全局阈值+中位数过滤(已调优)", mb)
        draw(axes[row, 2], page, gts, pa, "新版:列切分+无丢弃谷值聚类", ma)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "layout_bench.png"), dpi=115)
    with open(os.path.join(OUT, "layout_bench.json"), "w", encoding="utf8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    tot_gt = sum(results[m]["baseline_best_tuned"]["gt"] for m in results)
    tot_b = sum(results[m]["baseline_best_tuned"]["covered"] for m in results)
    tot_a = sum(results[m]["adaptive"]["covered"] for m in results)
    print(f"\nTOTAL rows {tot_gt}: baseline(best) {tot_b} ({tot_b/tot_gt:.1%}) | adaptive {tot_a} ({tot_a/tot_gt:.1%})")
    print("wrote", os.path.join(OUT, "layout_bench.png"))


if __name__ == "__main__":
    main()
