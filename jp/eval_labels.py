"""读标注网页导出的 glyph_labels.json，算真实准确率。

用法：
    python -m jp.eval_labels                      # 自动找 Downloads / out 目录
    python -m jp.eval_labels path\to\glyph_labels.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional


def find_file(arg: str = "") -> str:
    if arg and os.path.exists(arg):
        return arg
    cand: List[str] = []
    home = os.path.expanduser("~")
    for d in (os.path.join(home, "Downloads"), os.path.join(home, "下载"),
              os.path.join("out", ""), ".", os.path.join(home, "Desktop"),
              os.path.join(home, "桌面")):
        cand += glob.glob(os.path.join(d, "glyph_labels*.json"))
    if not cand:
        sys.exit("找不到 glyph_labels.json，请把导出文件放到 out\\ 下再运行")
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    path = find_file(argv[0] if argv else "")
    data = json.load(open(path, encoding="utf8"))
    rows = data.get("labels") or data.get("rows") or []
    print("标注文件:", path)
    print("样本:", len(rows), "\n")

    per: Dict[str, Dict[str, int]] = defaultdict(lambda: dict(n=0, ok=0, skip=0,
                                                              unknown=0, occhit=0))
    conf: Counter = Counter()
    octstat = Counter()
    for r in rows:
        page = r.get("page", "?")
        lab = r.get("label")
        st = per[page]
        if lab in (None, "", "skip"):
            st["skip"] += 1
            continue
        st["n"] += 1
        lab = str(lab)
        pred = str(r.get("pred", ""))
        if pred == lab:
            st["ok"] += 1
        else:
            conf[(pred, lab)] += 1
        if not pred.isdigit():
            st["unknown"] += 1
        if str(r.get("pred_oct", 0)) in ("0", ""):
            st["occhit"] += 1
        octstat[str(r.get("pred_oct"))] += 1

    tot_n = tot_ok = tot_skip = 0
    print(f"{'曲目':<14}{'已标':>6}{'猜对':>6}{'准确率':>9}{'跳过':>6}{'pred非数字':>11}")
    for page, st in per.items():
        acc = 100.0 * st["ok"] / max(1, st["n"])
        print(f"{page:<14}{st['n']:>6}{st['ok']:>6}{acc:>8.1f}%{st['skip']:>6}{st['unknown']:>11}")
        tot_n += st["n"]
        tot_ok += st["ok"]
        tot_skip += st["skip"]
    if tot_n:
        print(f"{'合计':<14}{tot_n:>6}{tot_ok:>6}"
              f"{100.0*tot_ok/tot_n:>8.1f}%{tot_skip:>6}")

    if conf:
        print("\n最常错的 15 组（系统猜 → 真值，次数）：")
        for (p, t), c in conf.most_common(15):
            print(f"   {p or '?':>2} → {t:<2}  {c}")
    print("\n八度分布（几何给出的 pred_oct）：", dict(sorted(octstat.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
