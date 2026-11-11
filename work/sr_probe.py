# -*- coding: utf-8 -*-
"""补充实验：
1) 与 bicubic / OpenCV-Cubic / bicubic+USM 对比（有无真实增益）；
2) 把 SRCNN 输出仿射对齐到 bicubic 后再比（增益是否只是对比度）；
3) 二值化后的连通块统计（笔画是否变粗/粘连）。
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = r"D:\简谱转调项目"
sys.path.insert(0, ROOT)
import jp.sr as sr  # noqa: E402

SRC = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
OUT = os.path.join(ROOT, "out")


def ych(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def lapv(y):
    return float(cv2.Laplacian(y, cv2.CV_32F, ksize=3).var())


def grad(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, 3)
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
    blur = cv2.GaussianBlur(y, (0, 0), sigma)
    return np.clip(y + amount * (y - blur), 0, 255)


def bin_stats(gray_u8, box, label):
    """box: (x0,y0,x1,y1) 在 1x 源图空间；返回该区域二值图的笔画统计。"""
    g = gray_u8
    sub = g[box[1]:box[3], box[0]:box[2]]
    _, bw = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    ink = (bw > 0)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    # 数字笔画宽度 ~ 2*P90(距离变换)，排除零星噪声
    vals = dist[ink]
    stroke = float(2 * np.percentile(vals, 90)) if vals.size else 0.0
    return {
        "label": label,
        "cc": int(n - 1),
        "ink_pct": round(float(ink.mean() * 100), 2),
        "stroke_px": round(stroke, 2),
        "mean_ink_val": round(float(sub[ink].mean()), 1) if ink.any() else 0.0,
    }


def main():
    bgr = sr.read_bgr(SRC)
    H, W = bgr.shape[:2]
    hh, ww = (H // 3) * 3, (W // 3) * 3
    hr = bgr[:hh, :ww]
    hr_y = ych(hr)
    lr = cv2.resize(hr, (ww // 3, hh // 3), interpolation=cv2.INTER_AREA)

    cands = {}
    cands["bicubic(PIL)"] = sr.bicubic(lr, 3)
    cands["cv2_cubic"] = cv2.resize(lr, (ww, hh), interpolation=cv2.INTER_CUBIC)
    cands["cv2_lanczos"] = cv2.resize(lr, (ww, hh), interpolation=cv2.INTER_LANCZOS4)
    z = sr.upscale(lr, 3)
    cands["SRCNN"] = z
    ub = cands["bicubic(PIL)"]
    yb = ych(ub)
    us = unsharp(yb, 1.2, 0.8)
    uh = np.clip(yb + 1.0 * (yb - cv2.GaussianBlur(yb, (0, 0), 1.5)), 0, 255)
    ys = ych(cands["SRCNN"])

    rows = {}
    hy = hr_y[:ys.shape[0], :ys.shape[1]]
    for name, img in cands.items():
        y = ych(img)[:hy.shape[0], :hy.shape[1]]
        rows[name] = {"psnr": round(psnr(hy, y), 2), "ssim": round(ssim(hy, y), 4),
                      "lap_var": round(lapv(y), 1), "grad": round(grad(y), 2),
                      "mean": round(float(y.mean()), 2), "std": round(float(y.std()), 2)}
    # 额外：bicubic(+USM) 两条基线
    for name, y in (("bicubic+USM(0.8)", us), ("bicubic+USM(1.0,s1.5)", uh)):
        y = y[:hy.shape[0], :hy.shape[1]]
        rows[name] = {"psnr": round(psnr(hy, y), 2), "ssim": round(ssim(hy, y), 4),
                      "lap_var": round(lapv(y), 1), "grad": round(grad(y), 2),
                      "mean": round(float(y.mean()), 2), "std": round(float(y.std()), 2)}
    rows["HR(真值)"] = {"psnr": None, "ssim": None, "lap_var": round(lapv(hy), 1),
                        "grad": round(grad(hy), 2), "mean": round(float(hy.mean()), 2),
                        "std": round(float(hy.std()), 2)}

    # 仿射对齐：y' = a*y + b 拟合到 bicubic，抵消亮度/对比度差
    yb2 = yb[:ys.shape[0], :ys.shape[1]]
    ys2 = ys[:ys.shape[0], :ys.shape[1]]
    A = np.stack([yb2.ravel(), np.ones(yb2.size)], 1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, ys2.ravel().astype(np.float64), rcond=None)
    ys_aff = np.clip(coef[0] * yb2 + coef[1], 0, 255).astype(np.float32)
    rows["SRCNN(仿射对齐到bicubic)"] = {
        "psnr": round(psnr(hy, ys_aff), 2), "ssim": round(ssim(hy, ys_aff), 4),
        "lap_var": round(lapv(ys_aff), 1), "grad": round(grad(ys_aff), 2),
        "mean": round(float(ys_aff.mean()), 2), "std": round(float(ys_aff.std()), 2),
        "affine_a_b": [round(float(coef[0]), 4), round(float(coef[1]), 4)],
    }

    # 二值化/笔画统计：源图第一条简谱行区域（1x 坐标）
    band = (40, 108, 460, 200)
    bstats = [
        bin_stats(cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY), band, "HR(1x)"),
        bin_stats(cv2.cvtColor(sr.bicubic(hr, 3), cv2.COLOR_BGR2GRAY),
                  (band[0] * 3, band[1] * 3, band[2] * 3, band[3] * 3), "bicubic(3x)"),
        bin_stats(cv2.cvtColor(sr.upscale(hr, 3), cv2.COLOR_BGR2GRAY),
                  (band[0] * 3, band[1] * 3, band[2] * 3, band[3] * 3), "SRCNN(3x)"),
    ]
    rep = {"recon_vs_HR": rows, "binarize": bstats, "band_1x": band}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    with open(os.path.join(OUT, "sr_probe.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
