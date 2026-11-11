"""Synthetic glyph dataset + training for the jianpu symbol classifier.

Why synthetic: the corpus has ~50 scores in a handful of fonts; a template matcher
dies on the ones it has never seen, and there is no labelled data at all.  Rendering
`0-7 # b other` in many fonts (and degrading them like print/scan/JPEG does) gives a
classifier that generalises across the corpus and - more importantly - has an explicit
`other` class, so a 汉字 can never be silently read as a note.
"""
from __future__ import annotations

import glob
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models")

CLASSES = list("01234567") + ["#", "b", "other"]
NORM = 32
INNER = 26

FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "ariali.ttf", "times.ttf", "timesbd.ttf", "timesi.ttf",
    "georgia.ttf", "georgiab.ttf", "calibri.ttf", "calibrib.ttf", "cambria.ttc",
    "consola.ttf", "constan.ttf", "BOOKOS.TTF", "BOOKOSB.TTF", "verdana.ttf", "verdanab.ttf",
    "tahoma.ttf", "palab.ttf", "simsun.ttc", "simhei.ttf", "msyh.ttc", "Deng.ttf", "Dengb.ttf",
    "STSONG.TTF", "STKAITI.TTF", "FRABK.TTF", "GARA.TTF", "CENTURY.TTF", "BAUHS93.TTF",
]

CJK = (
    "口日目回国田曲四品昌唱和吗知后号哥吕日月白自且在音曲谱音乐唱歌词一二三四五六七八九十"
    "我你他她它的了是不有这那大中小上下左右来回走爱心情春风花雨雪天地人家年时节分秒"
    "中华铄金娘傲娇系男孩告白克罗地亚狂想凉刀剑春秋前尘如梦勾指起誓同归世界线行健"
    "女为何踮脚尖尘梦忆仙怪物暴走万神纪世末歌者积雨云两小无猜暗杀教室片头三上下"
)
PUNCT = "，。、；：！？（）《》〈〉【】“”‘’·—…～-|/\\+*=%$@&"
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXTRA = "89()[]{},.:;!?"


def available_fonts() -> List[str]:
    out = []
    for name in FONT_CANDIDATES:
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            out.append(p)
    if not out:
        raise RuntimeError("no fonts found")
    return out


def _render(ch: str, font_path: str, size: int) -> np.ndarray:
    font = ImageFont.truetype(font_path, size)
    pad = size
    img = Image.new("L", (size * 3 + pad, size * 3 + pad), 255)
    d = ImageDraw.Draw(img)
    d.text((size, size), ch, fill=0, font=font)
    a = np.array(img)
    ink = (a < 160).astype(np.uint8) * 255
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return np.zeros((2, 2), np.uint8)
    return ink[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2]


def _lowres(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """模拟低分辨率栅格化：大字号渲染 → 缩到 10~17 px → 再放大回来。

    三月雨这种 742 px 的扫描件，数字只有 13 px，笔画被 JPEG 压成一两像素。
    只用 20-46 px 的清晰样本训练，模型在这种输入上会系统性读错。
    """
    h, w = ink.shape
    target = int(rng.integers(10, 18))
    sc = target / float(max(h, w))
    small = cv2.resize(ink, (max(2, int(round(w * sc))), max(2, int(round(h * sc)))),
                       interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        small = cv2.GaussianBlur(small, (3, 3), float(rng.uniform(0.4, 0.9)))
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def _augment(ink: np.ndarray, rng: np.random.Generator, hard: bool) -> np.ndarray:
    h, w = ink.shape
    # geometric jitter (print/scan)
    ang = rng.uniform(-3, 3) if hard else rng.uniform(-1.2, 1.2)
    shear = rng.uniform(-0.06, 0.06) if hard else rng.uniform(-0.02, 0.02)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 1] += shear
    ink = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    # stroke weight
    if rng.random() < 0.6:
        k = int(rng.integers(1, 3))
        ker = np.ones((k, k), np.uint8)
        ink = cv2.dilate(ink, ker) if rng.random() < 0.5 else cv2.erode(ink, ker)
    # blur + resample artefacts
    if rng.random() < 0.7:
        ink = cv2.GaussianBlur(ink, (3, 3), float(rng.uniform(0.3, 1.1)))
    # jpeg-ish noise
    if rng.random() < 0.5:
        noise = rng.normal(0, rng.uniform(3, 14), ink.shape)
        ink = np.clip(ink.astype(np.float32) + noise, 0, 255)
    # threshold jitter
    thr = rng.uniform(90, 170)
    ink = ((ink > thr) * 255).astype(np.uint8)
    return ink


def normalize(ink: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return np.zeros((NORM, NORM), np.float32)
    sub = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    s = INNER / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    res = cv2.resize(sub.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((NORM, NORM), np.uint8)
    oy, ox = (NORM - nh) // 2, (NORM - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = res
    return canvas.astype(np.float32) / 255.0


def _valid(ink: np.ndarray) -> bool:
    """Reject degenerate renders (tofu boxes, blank glyphs, destroyed by augmentation)."""
    if ink.size == 0:
        return False
    h, w = ink.shape
    if h < 5 or w < 2:
        return False
    ratio = float((ink > 0).mean())
    if ratio > 0.82 or ratio < 0.04:
        return False
    asp = w / float(h)
    if asp > 2.2 or asp < 0.12:
        return False
    return True


def sample(cls: str, fonts: List[str], rng: np.random.Generator, hard: bool = True) -> np.ndarray:
    for _ in range(6):
        font = fonts[int(rng.integers(0, len(fonts)))]
        size = int(rng.integers(20, 46))
        if cls in "01234567#b":
            ch = cls
        else:
            pool = CJK + PUNCT + LATIN + EXTRA
            ch = pool[int(rng.integers(0, len(pool)))]
        try:
            ink = _render(ch, font, size)
        except Exception:
            ink = _render(cls if cls != "other" else "口", fonts[0], size)
        if ink.shape[0] < 3 or ink.shape[1] < 3:
            continue
        if rng.random() < 0.45:
            ink = _lowres(ink, rng)
        ink = _augment(ink, rng, hard)
        if _valid(ink):
            return normalize(ink)
    return normalize(_render("8", fonts[0], 28))


def build(n_per_class: int, seed: int = 0, font_subset=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    fonts = font_subset or available_fonts()
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(sample(cls, fonts, rng))
            y.append(ci)
    return np.stack(X)[:, None, :, :], np.array(y, dtype=np.int64), fonts


# ---------------------------------------------------------------- model
def build_model():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 32),      # 32 -> 16
        block(32, 64),     # 16 -> 8
        block(64, 96),     # 8 -> 4
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(96, 96), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Linear(96, len(CLASSES)),
    )


def main():
    import argparse

    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(MODEL_DIR, "glyph_cnn.pt"))
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fonts = available_fonts()
    print(f"{len(fonts)} fonts", flush=True)

    # hold out 3 fonts for a generalisation check
    held = [fonts[i] for i in (0, len(fonts) // 2, len(fonts) - 1)]
    train_fonts = [f for f in fonts if f not in held]
    print("held out:", [os.path.basename(f) for f in held], flush=True)

    Xtr, ytr, _ = build(args.per_class, seed=args.seed, font_subset=train_fonts)
    print("train", Xtr.shape, flush=True)
    Xva, yva, _ = build(max(150, args.per_class // 8), seed=args.seed + 1, font_subset=held)
    print("held-out fonts val", Xva.shape, flush=True)

    dev = "cpu"
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)
    yva_t = torch.from_numpy(yva)
    n = len(Xtr_t)
    bs = 256
    best = 0.0
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        model.train()
        tot = cor = 0
        run = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            run += float(loss) * len(idx)
            cor += int((out.argmax(1) == yb).sum())
            tot += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1)
            acc = float((pred == yva_t).float().mean())
        print(f"ep{ep + 1:02d} loss={run / tot:.4f} train_acc={cor / tot:.4f} heldout_acc={acc:.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save({"state": model.state_dict(), "classes": CLASSES, "norm": NORM,
                        "held_out": [os.path.basename(f) for f in held]}, args.out)
            print("  saved", flush=True)
    print("best held-out acc", round(best, 4), "->", args.out)


if __name__ == "__main__":
    main()
