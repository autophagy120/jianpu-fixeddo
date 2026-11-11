"""Benchmark: what happens when a digit-only classifier is asked about 汉字.

The 2018 program classifies every candidate box against digit prototypes with a
1-NN rule and no abstention, so any closed-structure Hanzi (口日目回国田曲...) is
forced into 0/8/6/9.  This script quantifies that, and then measures how much
each guard recovers.

Run:  python bench_glyph.py
Out:  out/glyph_bench.json, out/glyph_bench.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jianpu import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT_DIGIT = r"C:\Windows\Fonts\arial.ttf"
FONT_HANZI = [r"C:\Windows\Fonts\simhei.ttf", os.path.join(os.path.dirname(HERE), "work", "extracted", "JianPu-Easy-Trans", "msyh.ttf")]
DIGITS = list("01234567")
# Hanzi that a jianpu page actually contains and that look like digits
HANZI_CLOSED = list("口日目回国田曲四品昌唱和吗知后号哥吕回曰")
SYMBOLS = list("#b()|:-~.")
SIZES = [24, 32, 42, 56]


def render(ch: str, font_path: str, size: int, rng: np.random.Generator) -> np.ndarray:
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (size * 3, size * 3), 255)
    d = ImageDraw.Draw(img)
    d.text((size // 2, size // 2), ch, fill=0, font=font)
    a = np.array(img)
    ink = (a < 128).astype(np.uint8) * 255
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return ink
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # scan / print noise
    if rng.random() < 0.5:
        import cv2

        s = float(rng.uniform(0.4, 0.9))
        ink = cv2.GaussianBlur(ink, (3, 3), s)
        ink = ((ink > 110) * 255).astype(np.uint8)
    if rng.random() < 0.3:
        import cv2

        k = int(rng.integers(1, 3))
        ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
    return ink


def feats_from_patch(ink: np.ndarray) -> core.GlyphFeatures:
    pad = np.pad(ink, 1)
    ys, xs = np.nonzero(pad)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    area = int((ink > 0).sum())
    comp = core.Comp(int(xs.min()), int(ys.min()), int(w), int(h), area)
    return core.glyph_features(pad.astype(np.uint8), comp)


def build_set(rng):
    rows = []
    for ch in DIGITS:
        for size in SIZES:
            ink = render(ch, FONT_DIGIT, size, rng)
            rows.append(("digit", ch, ink))
    for ch in HANZI_CLOSED:
        for fp in FONT_HANZI:
            for size in (32, 42):
                ink = render(ch, fp, size, rng)
                rows.append(("hanzi", ch, ink))
    for ch in SYMBOLS:
        ink = render(ch, FONT_DIGIT, 36, rng)
        rows.append(("symbol", ch, ink))
    return rows


def evaluate(rows, matcher: core.NearestTemplate):
    """Returns per-category stats for the three decision policies."""
    pol = {
        "1-NN 无拒识(现状)": dict(reject=False),
        "+形状拒识(IoU/间距/孔洞/宽高比)": dict(reject=True),
    }
    # ablation variants
    abl = {
        "仅 IoU 阈值": dict(reject=True, min_margin=0.0, max_holes=99, aspect_max=9.9),
        "仅 第一二名间距": dict(reject=True, min_iou=0.0, max_holes=99, aspect_max=9.9),
        "仅 孔洞/宽高比": dict(reject=True, min_iou=0.0, min_margin=0.0),
    }
    out = {}
    for name, kw in pol.items():
        stats = {"digit_hit": 0, "digit_tot": 0, "digit_emit": 0, "digit_emit_right": 0,
                 "hanzi_tot": 0, "hanzi_emit": 0, "sym_tot": 0, "sym_emit": 0}
        for kind, ch, ink in rows:
            f = feats_from_patch(ink)
            label, score = matcher.predict(f, **kw)
            if kind == "digit":
                stats["digit_tot"] += 1
                if label is not None:
                    stats["digit_emit"] += 1
                    stats["digit_emit_right"] += int(label == ch)
                    stats["digit_hit"] += int(label == ch)
            elif kind == "hanzi":
                stats["hanzi_tot"] += 1
                stats["hanzi_emit"] += int(label is not None)
            else:
                stats["sym_tot"] += 1
                stats["sym_emit"] += int(label is not None)
        out[name] = stats
    out["_ablation"] = {}
    for name, kw in abl.items():
        hanzi_emit = 0
        digit_hit = 0
        digit_tot = 0
        for kind, ch, ink in rows:
            f = feats_from_patch(ink)
            label, _ = matcher.predict(f, **kw)
            if kind == "hanzi":
                hanzi_emit += int(label is not None)
            elif kind == "digit":
                digit_tot += 1
                digit_hit += int(label == ch)
        out["_ablation"][name] = {"hanzi_false_accept": hanzi_emit, "digit_acc": digit_hit / max(1, digit_tot)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = build_set(rng)
    matcher = core.NearestTemplate()
    res = evaluate(rows, matcher)

    def summarize(s):
        digit_acc = s["digit_hit"] / max(1, s["digit_tot"])
        hanzi_fa = s["hanzi_emit"] / max(1, s["hanzi_tot"])
        sym_fa = s["sym_emit"] / max(1, s["sym_tot"])
        prec = s["digit_emit_right"] / max(1, s["digit_emit"])
        return {
            "digit_acc(强制作答)": round(digit_acc, 4),
            "digit_emitted": s["digit_emit"],
            "digit_emitted_correct_ratio": round(prec, 4),
            "hanzi_false_accept": s["hanzi_emit"],
            "hanzi_false_accept_rate": round(hanzi_fa, 4),
            "symbol_false_accept_rate": round(sym_fa, 4),
        }

    report = {}
    print(f"测试集: {len(DIGITS)*len(SIZES)} 数字 / {len(HANZI_CLOSED)*len(FONT_HANZI)*2} 汉字 / {len(SYMBOLS)} 符号\n")
    for name in ("1-NN 无拒识(现状)", "+形状拒识(IoU/间距/孔洞/宽高比)"):
        report[name] = summarize(res[name])
        r = report[name]
        print(f"{name}:")
        print(f"   数字准确率(必答)={r['digit_acc(强制作答)']:.3f}  数字被输出={r['digit_emitted']}/{len(DIGITS)*len(SIZES)}"
              f"  其中正确={r['digit_emitted_correct_ratio']:.3f}")
        print(f"   汉字被误判成数字: {r['hanzi_false_accept']}/{len(HANZI_CLOSED)*len(FONT_HANZI)*2} "
              f"= {r['hanzi_false_accept_rate']:.1%}   符号误判率={r['symbol_false_accept_rate']:.1%}")
    report["消融"] = res["_ablation"]
    print("\n消融(汉字误判数 / 数字准确率):")
    for k, v in res["_ablation"].items():
        print(f"   {k}: {v['hanzi_false_accept']} / {v['digit_acc']:.3f}")

    with open(os.path.join(OUT, "glyph_bench.json"), "w", encoding="utf8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # figure: score distributions
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rows_by = {"digit": [], "hanzi": [], "symbol": []}
    for kind, ch, ink in rows:
        f = feats_from_patch(ink)
        sc = matcher.scores(f)
        rows_by[kind].append(float(sc.max()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0, 1, 30)
    for kind, color, lab in (("digit", "#2ca02c", "数字 0-7"), ("hanzi", "#d62728", "歌词汉字"), ("symbol", "#7f7f7f", "谱面符号")):
        ax[0].hist(rows_by[kind], bins=bins, alpha=0.6, label=lab, color=color)
    ax[0].axvline(0.55, color="k", ls="--", lw=1)
    ax[0].text(0.56, ax[0].get_ylim()[1] * 0.9, "IoU 拒识阈值", fontsize=8)
    ax[0].set_xlabel("best template IoU (越大越像数字)")
    ax[0].set_ylabel("样本数")
    ax[0].set_title("最佳模板得分分布：数字 vs 歌词汉字")
    ax[0].legend(fontsize=8)

    names = ["1-NN 无拒识(现状)", "+形状拒识(IoU/间距/孔洞/宽高比)"]
    fa = [report[n]["hanzi_false_accept_rate"] for n in names]
    acc = [report[n]["digit_emitted_correct_ratio"] for n in names]
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [a * 100 for a in fa], 0.38, label="汉字被当成数字的比例", color="#d62728")
    ax[1].bar(x + 0.2, [a * 100 for a in acc], 0.38, label="已输出数字的正确率", color="#2ca02c")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["现状", "加拒识"], fontsize=9)
    ax[1].set_ylabel("%")
    for i, v in enumerate(fa):
        ax[1].text(i - 0.2, v * 100 + 1, f"{v*100:.0f}%", ha="center", fontsize=9)
    for i, v in enumerate(acc):
        ax[1].text(i + 0.2, v * 100 + 1, f"{v*100:.0f}%", ha="center", fontsize=9)
    ax[1].set_title("闭合结构汉字被误识别为数字的比例")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "glyph_bench.png"), dpi=130)
    print("\nwrote", os.path.join(OUT, "glyph_bench.png"))


if __name__ == "__main__":
    main()
