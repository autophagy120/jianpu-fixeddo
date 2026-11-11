# -*- coding: utf-8 -*-
"""jp.sr (SRCNN zoom=3) 真图测试：全页 3x 输出 + bicubic/SRCNN 对照图 + 数值对比。"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = r"D:\简谱转调项目"
sys.path.insert(0, ROOT)

import jp.sr as sr  # noqa: E402

SRC = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
OUT = os.path.join(ROOT, "out")
FONT = r"C:\Windows\Fonts\msyh.ttc"


# ------------------------------------------------------------------ metrics
def ychannel(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def lap_var(y: np.ndarray) -> float:
    """拉普拉斯响应方差：越大越锐（也对噪声敏感）。"""
    return float(cv2.Laplacian(y, cv2.CV_32F, ksize=3).var())


def grad_mean(y: np.ndarray) -> float:
    """平均梯度幅值（Sobel），等价于"边缘有多强"。"""
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def tenengrad(y: np.ndarray) -> float:
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


def psnr(a: np.ndarray, b: np.ndarray, crop: int = 4) -> float:
    if crop:
        a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(255.0 * 255.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray, crop: int = 4) -> float:
    try:
        from skimage.metrics import structural_similarity as f
    except Exception:
        return float("nan")
    if crop:
        a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    return float(f(a, b, data_range=255))


def stats(y: np.ndarray) -> dict:
    return {
        "lap_var": round(lap_var(y), 2),
        "grad_mean": round(grad_mean(y), 3),
        "tenengrad": round(tenengrad(y), 1),
        "clip255_pct": round(float((y >= 255).mean() * 100), 3),
        "clip0_pct": round(float((y <= 0).mean() * 100), 3),
        "mean": round(float(y.mean()), 2),
        "std": round(float(y.std()), 2),
    }


# ------------------------------------------------------------- crop picking
def digit_box(bgr: np.ndarray, win=(300, 120)) -> tuple[int, int, int, int]:
    """在简谱数字密集处自动挑一块 crop（源图坐标）: x, y, w, h。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    cand = [
        (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]), int(cen[i][0]), int(cen[i][1]))
        for i in range(1, n)
        if 5 <= st[i, 3] <= 16 and 2 <= st[i, 2] <= 16 and st[i, 4] >= 8
    ]
    W, H = win
    h, w = g.shape
    best, best_cnt = None, -1
    for y0 in range(0, max(1, h - H), 10):
        for x0 in range(0, max(1, w - W), 10):
            c = sum(1 for (_, _, _, _, cx, cy) in cand if x0 <= cx < x0 + W and y0 <= cy < y0 + H)
            if c > best_cnt:
                best, best_cnt = (x0, y0), c
    x0, y0 = best
    return int(x0), int(y0), int(W), int(H), best_cnt


# ------------------------------------------------------------------- labels
def label_strip(width: int, height: int, left: str, right: str, bg=255, fg=20) -> np.ndarray:
    """用 PIL 画一行中文标签（cv2.putText 不支持中文）。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    size = max(11, int(height * 0.52))
    try:
        font = ImageFont.truetype(FONT, size)
    except Exception:
        font = ImageFont.load_default()
    mid = width // 2
    d.text((8, (height - size) // 2 - 1), left, fill=(fg, fg, fg), font=font)
    tw = d.textlength(right, font=font) if hasattr(d, "textlength") else len(right) * size
    d.text((width - tw - 8, (height - size) // 2 - 1), right, fill=(fg, fg, fg), font=font)
    d.line([(mid - 2, 0), (mid - 2, height)], fill=(180, 180, 180), width=1)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print(json.dumps(sr.describe(), ensure_ascii=False, indent=2))

    bgr = sr.read_bgr(SRC)
    H, W = bgr.shape[:2]
    print("源图 %dx%d" % (W, H))

    # ---------------- 1) 全页 3x
    t0 = time.perf_counter()
    up = sr.upscale(bgr, 3)
    t_sr = time.perf_counter() - t0
    print("SRCNN 3x -> %r  耗时 %.2f s (net %.2f s, %d tiles)"
          % (up.shape, t_sr, sr.LAST["net_seconds"], sr.LAST["tiles"]))

    t0 = time.perf_counter()
    up_bi = sr.bicubic(bgr, 3)
    t_bi = time.perf_counter() - t0
    print("bicubic 3x -> %r 耗时 %.3f s" % (up_bi.shape, t_bi))

    p_jpg = os.path.join(OUT, "sr_test.jpg")
    sr.write_bgr(p_jpg, up, quality=95)
    print("写出", p_jpg, "%.1f MB" % (os.path.getsize(p_jpg) / 2 ** 20))

    # 第 2 次调用（模型已缓存）的耗时
    t0 = time.perf_counter()
    _ = sr.upscale(bgr, 3)
    t_sr2 = time.perf_counter() - t0

    # ---------------- 2) 对照图
    x0, y0, cw, ch, cnt = digit_box(bgr)
    print("自动选块 x=%d y=%d %dx%d 数字候选=%d" % (x0, y0, cw, ch, cnt))
    L = up_bi[y0 * 3:(y0 + ch) * 3, x0 * 3:(x0 + cw) * 3]
    R = up[y0 * 3:(y0 + ch) * 3, x0 * 3:(x0 + cw) * 3]
    assert L.shape == R.shape

    pad, sep = 10, 4
    ph, pw = L.shape[:2]
    lm = stats(ychannel(L))
    rm = stats(ychannel(R))
    head = label_strip(2 * pw + sep + 2 * pad, 34, "左：双三次插值 x3（bicubic）",
                       "右：SRCNN x3（仅亮度）", bg=248)
    body = np.full((ph, 2 * pw + sep + 2 * pad, 3), 255, np.uint8)
    body[0:ph, pad:pad + pw] = L
    body[0:ph, pad + pw:pad + pw + sep] = (70, 70, 220)      # 分隔线
    body[0:ph, pad + pw + sep:pad + 2 * pw + sep] = R
    cap_txt = ("边缘锐度(拉普拉斯方差)  bicubic %.0f -> SRCNN %.0f   "
               "平均梯度幅值  %.2f -> %.2f   零/饱和裁剪 %.2f%%"
               % (lm["lap_var"], rm["lap_var"], lm["grad_mean"], rm["grad_mean"], rm["clip255_pct"]))
    foot = label_strip(2 * pw + sep + 2 * pad, 30, cap_txt, "", bg=248)
    mont = np.vstack([head, body, foot])
    p_cmp = os.path.join(OUT, "sr_cmp.png")
    sr.write_bgr(p_cmp, mont)
    print("写出", p_cmp, "%.2f MB" % (os.path.getsize(p_cmp) / 2 ** 20))

    # ---------------- 3) 数值：全页 + crop
    rep = {
        "src": [W, H],
        "sr_seconds_total": round(t_sr, 3),
        "sr_seconds_warm": round(t_sr2, 3),
        "sr_seconds_net_only": round(float(sr.LAST["net_seconds"]), 3),
        "bicubic_seconds": round(t_bi, 3),
        "tiles": int(sr.LAST["tiles"]),
        "full_page": {"bicubic": stats(ychannel(up_bi)), "srcnn": stats(ychannel(up))},
        "crop": {"box": [x0, y0, cw, ch], "bicubic": lm, "srcnn": rm},
        "ratio": {
            "lap_var": round(rm["lap_var"] / max(lm["lap_var"], 1e-9), 3),
            "grad_mean": round(rm["grad_mean"] / max(lm["grad_mean"], 1e-9), 3),
        },
        "bicubic_vs_srcnn_psnr": round(psnr(ychannel(up_bi), ychannel(up)), 2),
    }

    # ---------------- 4) 自洽 SR 实验：以原页为 HR，缩 3 倍再放大，看谁更接近 HR
    hh, ww = (H // 3) * 3, (W // 3) * 3
    hr_bgr = bgr[:hh, :ww]
    hr_y = ychannel(hr_bgr)
    lr_bgr = cv2.resize(hr_bgr, (ww // 3, hh // 3), interpolation=cv2.INTER_AREA)
    up2_srcnn = sr.upscale(lr_bgr, 3)
    up2_bic = sr.bicubic(lr_bgr, 3)
    lr_y_up = ychannel(cv2.resize(lr_bgr, (ww, hh), interpolation=cv2.INTER_NEAREST))
    hr_y2 = ychannel(hr_bgr)[: up2_srcnn.shape[0], : up2_srcnn.shape[1]]
    r_bi = ychannel(up2_bic)[: hr_y2.shape[0], : hr_y2.shape[1]]
    r_sn = ychannel(up2_srcnn)[: hr_y2.shape[0], : hr_y2.shape[1]]
    rep["recon"] = {
        "note": "原页当 HR，INTER_AREA 缩 3 倍成 LR，再 3x 放大，与原页比",
        "hr_size": list(hr_y2.shape[::-1]),
        "bicubic": {"psnr": round(psnr(hr_y2, r_bi), 2), "ssim": round(ssim(hr_y2, r_bi), 4),
                    "lap_var": round(lap_var(r_bi), 1), "grad_mean": round(grad_mean(r_bi), 3)},
        "srcnn": {"psnr": round(psnr(hr_y2, r_sn), 2), "ssim": round(ssim(hr_y2, r_sn), 4),
                  "lap_var": round(lap_var(r_sn), 1), "grad_mean": round(grad_mean(r_sn), 3)},
        "hr": {"lap_var": round(lap_var(hr_y2), 1), "grad_mean": round(grad_mean(hr_y2), 3)},
        "lr_nearest_up": {"lap_var": round(lap_var(lr_y_up), 1),
                          "grad_mean": round(grad_mean(lr_y_up), 3)},
    }
    sr.write_bgr(os.path.join(OUT, "_sr_recon_hr.jpg"), hr_bgr, quality=95)
    sr.write_bgr(os.path.join(OUT, "_sr_recon_srcnn.png"), up2_srcnn)
    sr.write_bgr(os.path.join(OUT, "_sr_recon_bic.png"), up2_bic)

    p_json = os.path.join(OUT, "sr_metrics.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("写出", p_json)


if __name__ == "__main__":
    main()
