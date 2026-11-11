"""把人工标注网页导出的 JSON 写回模板库。

用法：
    # 把网页里"导出 JSON"的内容存成 out/labels.json
    python -m jp.apply_labels out/labels.json
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np

from . import templbank as TB


def main(path: str):
    got = TB.load()
    if not got:
        raise SystemExit("还没有模板库")
    cs, labels, counts = got
    with open(path, encoding="utf8") as fh:
        mapping: Dict[str, str] = json.load(fh)

    changed = 0
    for k, v in mapping.items():
        i = int(k)
        v = str(v).strip()
        if v not in ("other",) and v not in "01234567":
            continue
        if 0 <= i < len(labels) and labels[i] != v:
            labels[i] = v
            changed += 1
    np.savez_compressed(TB.BANK_PATH, centroids=cs, labels=np.array(labels), counts=counts)

    named = [(i, l, int(c)) for i, (l, c) in enumerate(zip(labels, counts)) if l in "01234567"]
    cov = sum(c for _i, _l, c in named) / max(1, counts.sum())
    print(f"写回 {changed} 条；已命名 {len(named)} 簇，覆盖 {cov:.1%} 的字形实例")
    for d in "01234567":
        n = sum(1 for _i, l, _c in named if l == d)
        c = sum(c for _i, l, c in named if l == d)
        print(f"  数字 {d}: {n} 簇 / {c} 例")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/labels.json")
