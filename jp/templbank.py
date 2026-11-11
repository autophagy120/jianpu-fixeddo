"""语料级模板库：在**原始分辨率**上收集全语料字形，聚类成少量紧密模板簇。

为什么这样做（而不是训练大模型）：
- 用户的谱来自同一批出版物，字体只有 2-3 种；同一字体同一字号下，同一个数字的位图
  几乎像素级重合（实测簇内 IoU 0.94-1.00）；
- easytr 的 1009 个原型干的就是这件事，只是它的特征定义在编译产物里拿不到；
- 所以我们用**用户自己的数据**建同等作用的模板库：确定、离线、可增补。

产物：models/templ_bank.npz {centroids(uint8 NxHxW), labels(str N)}

用法：
    python -m jp.templbank build   # 扫语料 + 聚类 + 出代表图（等人工/VLM 命名）
    python -m jp.templbank label   # 用 VLM 给簇命名（分小批 + 重试 + 唯一性约束）
    python -m jp.templbank stats   # 看看库的大小/覆盖
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import core

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models")
BANK_PATH = os.path.join(MODEL_DIR, "templ_bank.npz")
CORPUS = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱"

MASK = 20           # 统一画布
INNER = 16


# ------------------------------------------------------------------ 形状工具
def norm_mask(patch: np.ndarray, size: int = MASK, inner: int = INNER) -> np.ndarray:
    a = patch if patch.dtype == np.uint8 else patch.astype(np.uint8)
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return np.zeros((size, size), np.uint8)
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    s = inner / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    res = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
    can = np.zeros((size, size), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    can[oy:oy + nh, ox:ox + nw] = (res > 100).astype(np.uint8)
    return can


def clean(patch: np.ndarray) -> np.ndarray:
    """剥掉贴着的细横/细竖线（减时线、连音线残片）和小噪点，再裁到内容。"""
    a = patch if patch.dtype == np.uint8 else patch.astype(np.uint8)
    if a.size == 0:
        return a
    thin = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    thin_v = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    out = cv2.bitwise_and(a, cv2.bitwise_not(cv2.bitwise_or(thin, thin_v)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    for i in range(1, n):
        if stats[i, 4] < 4:
            out[lab == i] = 0
    ys, xs = np.nonzero(out)
    if len(ys) == 0:
        return a
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    A, B = a > 0, b > 0
    u = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum()) / u if u else 0.0


@dataclass
class Bucket:
    mask: np.ndarray
    count: int = 1
    members: int = 0

    @property
    def key(self) -> bytes:
        return self.mask.tobytes()


# ------------------------------------------------------------------ 采集
def page_glyphs(path: str) -> List[np.ndarray]:
    """原始分辨率下取一页的"像数字"字形（不做放大）。"""
    try:
        g = core.load_gray(path)
    except Exception:
        return []
    if g.shape[0] < 200 or g.shape[1] < 200:
        return []
    bw = core.binarize(g)
    comps = core.components(bw, min_area=4)
    if not comps:
        return []
    hs = np.array([c.h for c in comps if c.h >= 6 and 0.3 <= c.aspect <= 1.0])
    if len(hs) < 20:
        return []
    h_med = float(np.median(hs))
    lo, hi = max(5, 0.72 * h_med), 1.45 * h_med
    clean_bw, _ = core.strip_rules(bw, max(6, int(0.6 * h_med)), max(5, int(0.5 * h_med)))
    out = []
    for c in core.components(clean_bw, min_area=4):
        if not (lo <= c.h <= hi):
            continue
        if not (0.15 <= c.aspect <= 1.15):
            continue
        patch = clean_bw[c.y:c.y + c.h, c.x:c.x + c.w]
        if patch.size == 0:
            continue
        out.append(norm_mask(clean(patch)))
    return out


def collect(corpus: str = CORPUS, limit: Optional[int] = None) -> List[np.ndarray]:
    import re

    produced = re.compile(r"\(原调.*转调.*\)")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for root, _d, fs in os.walk(corpus):
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() in exts and not produced.search(f):
                files.append(os.path.join(root, f))
    if limit:
        files = files[:limit]
    masks: List[np.ndarray] = []
    for i, f in enumerate(files, 1):
        got = page_glyphs(f)
        masks.extend(got)
        print(f"  [{i}/{len(files)}] {os.path.basename(f)[:44]:44s} +{len(got):4d}  累计 {len(masks)}",
              flush=True)
    return masks


# ------------------------------------------------------------------ 聚类
def cluster(masks: Sequence[np.ndarray], sim: float = 0.82, min_count: int = 20) -> List[Bucket]:
    """先按 8x8 粗量化分桶（O(N)），再在桶代表之间做 IoU 合并。"""
    buckets: Dict[bytes, Bucket] = {}
    for m in masks:
        small = cv2.resize(m.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
        key = (small > 0.5).astype(np.uint8).tobytes()
        if key in buckets:
            buckets[key].count += 1
        else:
            buckets[key] = Bucket(mask=m)
    reps = sorted(buckets.values(), key=lambda b: -b.count)
    merged: List[Bucket] = []
    for b in reps:
        hit = None
        for c in merged:
            if iou(b.mask, c.mask) >= sim:
                hit = c
                break
        if hit is None:
            merged.append(b)
        else:
            hit.count += b.count
    merged.sort(key=lambda b: -b.count)
    total = sum(b.count for b in merged) or 1
    kept = [b for b in merged if b.count >= min_count]
    covered = sum(b.count for b in kept) / total
    print(f"分桶后 {len(reps)} 个代表 -> 合并成 {len(merged)} 簇；"
          f"计数≥{min_count} 的 {len(kept)} 簇覆盖 {covered:.1%}")
    return kept


# ------------------------------------------------------------------ 存取
def save(buckets: Sequence[Bucket], labels: Sequence[str], path: str = BANK_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cs = np.stack([b.mask for b in buckets]).astype(np.uint8)
    np.savez_compressed(path, centroids=cs, labels=np.array(list(labels)),
                        counts=np.array([b.count for b in buckets]))
    print(f"模板库写入 {path}  ({len(buckets)} 簇)")
    return path


def load(path: str = BANK_PATH):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return z["centroids"], [str(x) for x in z["labels"]], z["counts"]


# ------------------------------------------------------------------ CLI
def cmd_build(argv):
    corpus = argv[0] if argv else CORPUS
    limit = int(argv[1]) if len(argv) > 1 else None
    masks = collect(corpus, limit)
    print(f"共采集 {len(masks)} 个字形")
    buckets = cluster(masks)
    montage(buckets, os.path.join(ROOT, "out", "bank_montage.png"))
    save(buckets, ["?"] * len(buckets))


def montage(buckets: Sequence[Bucket], out_path: str, tile: int = 120, cols: int = 6, scale: int = 5):
    rows = int(np.ceil(len(buckets) / cols))
    canvas = np.full((rows * (tile + 44), cols * (tile + 14), 3), 255, np.uint8)
    for i, b in enumerate(buckets):
        r, c = divmod(i, cols)
        y0, x0 = r * (tile + 44) + 34, c * (tile + 14) + 7
        big = cv2.resize(255 - b.mask * 255, (tile, tile), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + tile + 1, y0 + tile + 1), (170, 170, 170), 1)
        cv2.putText(canvas, f"#{i} n={b.count}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 200), 1)
    canvas = cv2.resize(canvas, None, fx=scale / 5.0, fy=scale / 5.0, interpolation=cv2.INTER_CUBIC)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    core.imwrite(out_path, canvas)
    print(f"簇代表图写入 {out_path}（{len(buckets)} 个）")


def main(argv):
    cmd = argv[0] if argv else "build"
    if cmd == "build":
        cmd_build(argv[1:])
    elif cmd == "stats":
        got = load()
        if not got:
            print("还没有模板库")
            return
        cs, labels, counts = got
        print(f"{len(cs)} 簇，已命名 {sum(1 for x in labels if x in '01234567')} 个")
        for i, (l, c) in enumerate(zip(labels, counts)):
            print(f"  #{i:2d} {l:5s} {int(c)}")
    else:
        print("用法: python -m jp.templbank [build|stats]")


if __name__ == "__main__":
    main(sys.argv[1:])
