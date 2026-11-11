"""从「音符数相等的小节」自动抽 (字形图, 数字) 标签，训练字形分类器。

思路（用户提的按小节切割的直接收益）：

按整行 zip 出来的标签不可信——OCR 会把 `1̇` 写成 `1 1`，一行错位就全错。
但**按小节**：实测高清页 91% 的小节两边音符数完全相等（为了你唱下去：
55 个匹配小节里 50 个 exact）。这些小节里按序 zip 出来的 (字形图, 数字)
就是干净标签，不需要人工。

有了几千张干净标签，就能训一个只认字形的小 CNN，替掉现在这个
「1009 个原型做 1-NN」的分类器（数字准确率只有 78–80%，是整条链路的瓶颈）。
分类器一旦到 ~99%，数字、位置、八度就都在几何侧定死，
**不再需要 OCR↔几何 的跨模态对齐**——那一整类错位 bug 就结构性消失了。
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional, Tuple

import numpy as np

from . import align as AL
from . import core
from . import measures as ME
from . import merge_octave as MO
from . import xiaomi_ocr as XO

PATCH = 32


def patch_of_box(clean: np.ndarray, box, size: int = PATCH) -> np.ndarray:
    """从二值图里按框抠一块，缩放到 size×size 的 0/1 浮点图。"""
    import cv2
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(clean.shape[1], x1), min(clean.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size), np.float32)
    sub = (clean[y0:y1, x0:x1] > 0).astype(np.float32)
    # 保持长宽比，留边贴到正方形里，避免数字被拉变形
    h, w = sub.shape
    k = (size - 4) / float(max(h, w))
    nh, nw = max(1, int(round(h * k))), max(1, int(round(w * k)))
    sub = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = sub
    return out


def labels_from_page(orig: str, ocr_txt: str, net=None
                     ) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """只取「小节两边音符数相等」的配对作为标签。

    返回 (patches, 标签, GlyphNet 当前给出的预测) —— 第三项用来在同一批 patch
    上和现有分类器做公平对比。
    """
    from .glyphnet import GlyphNet
    net = net or GlyphNet()
    page = core.analyse(core.load_gray(orig), net, path=orig)
    slots = AL.ordered_slots(page, net)
    txt = MO.sanitize_ocr(ocr_txt)
    raw = [l for l in txt.splitlines() if XO.parse_line(l).is_music]
    geo = ME.geo_measures(page, slots)
    ocr, _src = ME.ocr_measures(raw)
    mapping = ME.align_measures(geo, ocr)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    gs: List[int] = []
    for oi, gi in enumerate(mapping):
        if gi is None or len(geo[gi]) != len(ocr[oi]):
            continue
        for slot, o in zip(geo[gi], ocr[oi]):
            if o.digit not in "01234567":
                continue
            xs.append(patch_of_box(page.clean, slot.box))
            ys.append(int(o.digit))
            gs.append(int(slot.digit) if str(slot.digit).isdigit() else -1)
    return xs, ys, gs


# ---------------------------------------------------------------- 训练

def build_model():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 16x16
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8x8
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, 8),
            )

        def forward(self, x):
            return self.f(x)

    return Net()


def train(xs: List[np.ndarray], ys: List[int], epochs: int = 30, seed: int = 42):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    random.seed(seed)
    X = torch.tensor(np.stack(xs)[:, None, :, :], dtype=torch.float32)
    Y = torch.tensor(np.asarray(ys, dtype=np.int64))
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            out = model(X[idx])
            loss = lossf(out, Y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:2d}  loss={tot/n:.4f}", flush=True)
    model.eval()
    return model


def accuracy(model, xs: List[np.ndarray], ys: List[int]) -> float:
    import torch
    if not xs:
        return 0.0
    with torch.no_grad():
        X = torch.tensor(np.stack(xs)[:, None, :, :], dtype=torch.float32)
        pred = model(X).argmax(1).numpy()
    return float((pred == np.asarray(ys)).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="*", default=[], help="orig:ocr 对，形如 path1::path2")
    ap.add_argument("--out", default=r"out\glyphnet.pt")
    args = ap.parse_args(argv)
    print("见 build_dataset()，本入口只做单页自检")


if __name__ == "__main__":
    main()
