"""读 dot_label.html 导出的 dot_labels.json，算八度点的真实准确率。

把网页里的编码还原成点数：
    0 = 无点 · 1 = 上 1 点 · 2 = 上 2 点 · 3 = 下 1 点 · 4 = 下 2 点
    x = 不是音符（应被排除）· skip = 跳过
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def find_file(arg: str = "") -> str:
    if arg and os.path.exists(arg):
        return arg
    home = os.path.expanduser("~")
    cand: List[str] = []
    for d in (os.path.join(home, "Downloads"), os.path.join(home, "下载"),
              "out", ".", os.path.join(home, "Desktop"), os.path.join(home, "桌面")):
        cand += glob.glob(os.path.join(d, "dot_labels*.json"))
    if not cand:
        sys.exit("找不到 dot_labels.json，请把它放到 out\\ 下再运行")
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def code_to_oct(v) -> Tuple[int, bool]:
    """返回 (几何意义上的点数, 是否是音符)。"""
    if v == "x":
        return 0, False
    return {"0": 0, "1": 1, "2": 2, "3": -1, "4": -2}.get(str(v), 0), True


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    path = find_file(argv[0] if argv else "")
    data = json.load(open(path, encoding="utf8"))
    rows = data.get("labels") or []
    print("标注文件:", path)
    print("样本:", len(rows), "\n")

    per: Dict[str, Dict[str, int]] = defaultdict(
        lambda: dict(n=0, ok=0, miss=0, extra=0, wrong_dir=0, notnote=0,
                     split=0, skip=0))
    confusion: Counter = Counter()
    total_missed = 0
    for r in rows:
        page = r.get("page", "?")
        v = r.get("label")
        st = per[page]
        if v in (None, "", "skip"):
            st["skip"] += 1
            continue
        truth, is_note = code_to_oct(v)
        if v == "s":
            st["split"] += 1
            continue
        if not is_note:
            st["notnote"] += 1
            continue
        st["n"] += 1
        pred = int(r.get("pred_oct", 0) or 0)
        if pred == truth:
            st["ok"] += 1
        else:
            confusion[(pred, truth)] += 1
            if pred == 0 and truth != 0:
                st["miss"] += 1
                total_missed += 1
            elif pred != 0 and truth == 0:
                st["extra"] += 1
            else:
                st["wrong_dir"] += 1

    hdr = f"{'曲目':<14}{'已标':>6}{'正确':>6}{'准确率':>9}{'漏点':>6}{'多点':>6}{'方向错':>8}{'非音符':>8}"
    print(hdr)
    tn = tok = 0
    for page, st in per.items():
        acc = 100.0 * st["ok"] / max(1, st["n"])
        print(f"{page:<14}{st['n']:>6}{st['ok']:>6}{acc:>8.1f}%"
              f"{st['miss']:>6}{st['extra']:>6}{st['wrong_dir']:>8}{st['notnote']:>8}")
        tn += st["n"]
        tok += st["ok"]
    if tn:
        print(f"{'合计':<14}{tn:>6}{tok:>6}{100.0*tok/tn:>8.1f}%")

    if confusion:
        print("\n错误明细（几何判断 → 真值，次数）：")
        lab = {0: "无点", 1: "上1", 2: "上2", -1: "下1", -2: "下2"}
        for (p, t), c in confusion.most_common(12):
            print(f"   {lab.get(p, p):>4} → {lab.get(t, t):<4} {c}")
    print(f"\n**漏点合计 {total_missed} 个** —— 这些是原谱有点、系统没看出来（最伤成品的一类）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
