"""End-to-end run on a real score (成都, published with the project).

Shows: column/row segmentation -> glyph decisions with and without abstention ->
per-row coverage report (the "silently missed row" guard).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jianpu import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SAMPLE = os.path.join(os.path.dirname(HERE), "work", "samples", "chengdu.jpg")


def analyse(path: str) -> Dict:
    gray = core.load_gray(path)
    bw = core.binarize(gray)
    clean, rules = core.strip_rules(bw)
    cols = core.find_columns(clean)
    matcher = core.NearestTemplate()

    rows = []
    for ci, (cx0, cx1) in enumerate(cols):
        for y0, y1 in core.split_adaptive(core.row_profile(clean[:, cx0:cx1])):
            band = clean[y0:y1, cx0:cx1]
            comps = [c for c in core.components(band, min_area=6) if c.h >= 6]
            if not comps:
                continue
            hs = np.array([c.h for c in comps])
            # glyphs much smaller than the band's dominant size are dots/punctuation
            main = comps
            base_labels, imp_labels, confs = [], [], []
            for c in main:
                f = core.glyph_features(band, c)
                lb, sb = matcher.predict(f, reject=False)
                li, si = matcher.predict(f, reject=True)
                base_labels.append(lb)
                imp_labels.append(li)
                confs.append(sb)
            n = len(main)
            digit_ratio_base = sum(1 for l in base_labels if l) / n
            digit_ratio_imp = sum(1 for l in imp_labels if l) / n
            rows.append({
                "col": ci,
                "y0": int(y0 + (0 if True else 0)),
                "y1": int(y1),
                "x0": int(cx0),
                "x1": int(cx1),
                "n_glyphs": n,
                "digit_emit_base": sum(1 for l in base_labels if l),
                "digit_emit_imp": sum(1 for l in imp_labels if l),
                "digit_ratio_base": round(digit_ratio_base, 3),
                "digit_ratio_imp": round(digit_ratio_imp, 3),
                "median_h": int(np.median(hs)),
                "kind": ("music" if (digit_ratio_imp >= 0.25 and n >= 8) else "lyric"),
                "labels_base": base_labels,
                "labels_imp": imp_labels,
                "conf": [round(c, 3) for c in confs],
                "comps": [(int(c.x) + cx0, int(c.y) + y0, int(c.w), int(c.h)) for c in main],
            })
    return {"shape": list(gray.shape), "columns": cols, "rows": rows, "gray": gray}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = analyse(SAMPLE)
    rows = res["rows"]
    print(f"page {res['shape']}, columns={res['columns']}, rows={len(rows)}\n")

    tot_base = sum(r["digit_emit_base"] for r in rows)
    tot_imp = sum(r["digit_emit_imp"] for r in rows)
    tot_glyph = sum(r["n_glyphs"] for r in rows)
    print(f"字形候选 {tot_glyph} 个;  无拒识输出数字 {tot_base} 个;  有拒识输出数字 {tot_imp} 个")

    lyric = [r for r in rows if r["kind"] == "lyric" and r["n_glyphs"] >= 15]
    music = [r for r in rows if r["kind"] == "music"]
    lb = sum(r["digit_emit_base"] for r in lyric)
    li = sum(r["digit_emit_imp"] for r in lyric)
    lg = sum(r["n_glyphs"] for r in lyric)
    print(f"判为歌词行的 {len(lyric)} 行、共 {lg} 个字形:  无拒识被迫输出数字 {lb} 个 ({lb/max(1,lg):.0%})"
          f"  ->  有拒识后 {li} 个 ({li/max(1,lg):.0%})")
    print(f"判为谱行的 {len(music)} 行")

    empty = [r for r in rows if r["digit_emit_imp"] == 0]
    print(f"\n覆盖率自检: {len(rows)-len(empty)}/{len(rows)} 行至少产出 1 个音符;  空行 {len(empty)} 个 -> 这些会被明确标为待确认，而不是静默消失")

    # ---- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    gray = res["gray"]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 8.6))
    for a in ax:
        a.imshow(gray, cmap="gray", vmin=0, vmax=255)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_title("原图 + 自适应行切分", fontsize=11)
    for r in rows:
        color = "#1f77b4" if r["kind"] == "music" else "#2ca02c"
        ax[0].add_patch(Rectangle((r["x0"], r["y0"]), r["x1"] - r["x0"], r["y1"] - r["y0"],
                                  fill=False, ec=color, lw=1.1))
        ax[0].text(r["x0"] + 3, r["y0"] + 4, f"{'谱' if r['kind']=='music' else '词'}", fontsize=7,
                   color=color)
    ax[1].set_title(f"现状式 1-NN（无拒识）：输出 {tot_base} 个数字", fontsize=11)
    ax[2].set_title(f"加形状拒识：输出 {tot_imp} 个数字", fontsize=11)
    for r in rows:
        for (cx, cy, cw, ch), lb, li, cf in zip(r["comps"], r["labels_base"], r["labels_imp"], r["conf"]):
            if lb:
                ax[1].add_patch(Rectangle((cx, cy), cw, ch, fill=False, ec="#d62728", lw=0.9))
            if li:
                ax[2].add_patch(Rectangle((cx, cy), cw, ch, fill=False, ec="#2ca02c", lw=0.9))
            elif lb:
                ax[2].add_patch(Rectangle((cx, cy), cw, ch, fill=False, ec="#ff7f0e", lw=0.9, ls=":"))
    ax[1].text(10, gray.shape[0] - 10, "红框 = 被当成音符的候选框（含歌词里的汉字）", fontsize=8,
               bbox=dict(fc="w", alpha=0.8))
    ax[2].text(10, gray.shape[0] - 10, "绿框 = 保留的音符   橙虚框 = 被拒识、转人工/大模型复核", fontsize=8,
               bbox=dict(fc="w", alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "real_overlay.png"), dpi=120)

    slim = {k: v for k, v in res.items() if k != "gray"}
    with open(os.path.join(OUT, "real_report.json"), "w", encoding="utf8") as fh:
        json.dump(slim, fh, ensure_ascii=False, indent=2)
    print("\nwrote", os.path.join(OUT, "real_overlay.png"), "and real_report.json")


if __name__ == "__main__":
    main()
