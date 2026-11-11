# -*- coding: utf-8 -*-
"""合成简谱行实验：3x 渲染当 GT -> /3 成 9px 数字（与真实页一致）-> 3x 放大后比 GT。
无 JPEG 伪影、尺度精确，是对 SRCNN 最有利的理想工况。"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = r"D:\简谱转调项目"
sys.path.insert(0, ROOT)
import jp.sr as sr  # noqa: E402

OUT = os.path.join(ROOT, "out")


def ych(b): return cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)


def lapv(y): return float(cv2.Laplacian(y, cv2.CV_32F, ksize=3).var())


def grad(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, 3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def psnr(a, b, crop=6):
    a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b, crop=6):
    from skimage.metrics import structural_similarity as f
    a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    return float(f(a, b, data_range=255))


def unsharp(y, sigma=1.2, amount=0.8):
    return np.clip(y + amount * (y - cv2.GaussianBlur(y, (0, 0), sigma)), 0, 255).astype(np.float32)


def render_line(scale: int, seed: int = 7) -> np.ndarray:
    """按 scale 倍渲染一行简谱：数字 + 减时值下划线 + 八度点 + 拍号分隔。"""
    rng = np.random.default_rng(seed)
    W, H = 620 * scale, 120 * scale
    img = np.full((H, W, 3), 255, np.uint8)
    base_y = int(66 * scale)
    fs = 1.05 * scale
    th = max(1, int(round(1.4 * scale)))
    x = 14 * scale
    F = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(int(6 * scale) if scale > 1 else 6):
        pass
    n_groups = 7
    for g in range(n_groups):
        for k in range(4):
            d = str(int(rng.integers(1, 8)))
            if rng.random() < 0.18:
                d = "i"
            (tw, tht), _ = cv2.getTextSize(d, F, fs, th)
            cv2.putText(img, d, (x, base_y), F, fs, (0, 0, 0), th, cv2.LINE_AA)
            # 减时值下划线
            if rng.random() < 0.45:
                y = base_y + int(4 * scale)
                cv2.line(img, (x, y), (x + tw, y), (0, 0, 0), max(1, int(round(1.0 * scale))))
            # 八度点
            if rng.random() < 0.22:
                cy = base_y - int(20 * scale)
                cv2.circle(img, (x + tw // 2, cy), max(1, int(round(1.2 * scale))), (0, 0, 0), -1)
            x += tw + int(4 * scale)
        x += int(12 * scale)
        if g < n_groups - 1:
            cv2.line(img, (x, base_y - int(26 * scale)), (x, base_y + int(6 * scale)),
                     (0, 0, 0), max(1, int(round(1.2 * scale))))   # 小节线
            x += int(14 * scale)
    return img


def main():
    gt = render_line(3)                                   # (360, 1860, 3) 3x 真值
    hh, ww = (gt.shape[0] // 3) * 3, (gt.shape[1] // 3) * 3
    gt = gt[:hh, :ww]
    lr = cv2.resize(gt, (ww // 3, hh // 3), interpolation=cv2.INTER_AREA)   # 9px 数字
    print("GT", gt.shape, "LR", lr.shape, "数字高度 ~%d px" % (70 // 3))

    cands = {"bicubic(PIL)": sr.bicubic(lr, 3),
             "cv2_cubic": cv2.resize(lr, (ww, hh), interpolation=cv2.INTER_CUBIC),
             "cv2_lanczos": cv2.resize(lr, (ww, hh), interpolation=cv2.INTER_LANCZOS4),
             "SRCNN": sr.upscale(lr, 3)}
    yb = ych(cands["bicubic(PIL)"])
    cands["bicubic+USM(0.8)"] = cv2.cvtColor(unsharp(yb, 1.2, 0.8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cands["bicubic+USM(1.0)"] = cv2.cvtColor(unsharp(yb, 1.5, 1.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    g_y = ych(gt)
    rows = {}
    for name, img in cands.items():
        y = ych(img)
        rows[name] = {"psnr": round(psnr(g_y, y), 2), "ssim": round(ssim(g_y, y), 4),
                      "lap_var": round(lapv(y), 1), "grad": round(grad(y), 2)}
        p = os.path.join(OUT, "_synth_%s.png" % name.replace("(", "_").replace(")", "").replace("+", "p"))
        sr.write_bgr(p, img)
    rows["GT"] = {"psnr": None, "ssim": None, "lap_var": round(lapv(g_y), 1), "grad": round(grad(g_y), 2)}

    # 二值图差：与 GT 的二值图 IoU / 数字块数
    def bw(y, inv=True):
        _, b = cv2.threshold(y.astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return b > 0

    gt_bw = bw(g_y)
    for name, img in cands.items():
        b = bw(ych(img))
        iou = float((b & gt_bw).sum() / max((b | gt_bw).sum(), 1))
        n1 = cv2.connectedComponents(gt_bw.astype(np.uint8))[0] - 1
        n2 = cv2.connectedComponents(b.astype(np.uint8))[0] - 1
        rows[name]["cc"] = int(n2)
        rows[name]["cc_gt"] = int(n1)
        rows[name]["iou_vs_GT"] = round(iou, 4)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    sr.write_bgr(os.path.join(OUT, "_synth_GT.png"), gt)
    sr.write_bgr(os.path.join(OUT, "_synth_LR_x3nearest.png"),
                 cv2.resize(lr, (ww, hh), interpolation=cv2.INTER_NEAREST))
    with open(os.path.join(OUT, "sr_synth.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # 对照拼图：GT | bicubic | SRCNN | USM
    tiles = [("GT(3x真值)", gt), ("bicubic", cands["bicubic(PIL)"]),
             ("SRCNN", cands["SRCNN"]), ("bicubic+USM", cands["bicubic+USM(0.8)"])]
    x0, y0, ch, cw = 0, 40, 200, 560
    fs = 0.7
    strips = []
    labels = []
    for name, img in tiles:
        crop = img[y0:y0 + ch, x0:x0 + cw].copy()
        strips.append(crop)
        labels.append(name)
    sep = np.full((ch, 3, 3), (60, 60, 60), np.uint8)
    row = []
    for i, (s, n) in enumerate(zip(strips, labels)):
        row.append(s)
        if i < len(strips) - 1:
            row.append(sep)
    body = np.hstack(row)
    head = np.full((26, body.shape[1], 3), 255, np.uint8)
    for i, n in enumerate(labels):
        cv2.putText(head, n, (i * (cw + 3) + 4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    sr.write_bgr(os.path.join(OUT, "sr_synth_cmp.png"), np.vstack([head, body]))
    print("写出", os.path.join(OUT, "sr_synth_cmp.png"))


if __name__ == "__main__":
    main()
