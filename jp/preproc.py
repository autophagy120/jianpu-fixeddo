"""预处理：把参考项目（arXiv:2512.14758）的 relight / deskew 接进来。

关键点：**不要让它迁就我的约定，而是让管线吃下它的输出**。
它返回的是「亮字暗底、0–1 float」，我的下游假设「暗字亮底、uint8 0–255」，
所以在入口处做一次极性与量纲归一化即可（不修改对方代码）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .refimg import deskew as ref_deskew
from .refimg import relighting as ref_relight


def to_uint8_ink_dark(img: np.ndarray) -> np.ndarray:
    """任意 dtype/极性 -> uint8，且保证「暗字亮底」。

    - float 0–1：先拉到 0–255；
    - 若大多数像素是暗的（即"亮字暗底"），整体反相。
    """
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        mx = float(a.max())
        if mx <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # 用中位数判断极性：纸张背景像素永远占多数
    if float(np.median(a)) < 128:
        a = 255 - a
    return a


def relight(gray: np.ndarray) -> Tuple[np.ndarray, dict]:
    """双伽马光照归一化。返回 (uint8 暗字亮底, 日志)"""
    try:
        out, log = ref_relight.adaptive_lighting_correction(gray)
        return to_uint8_ink_dark(out), (log if isinstance(log, dict) else {"log": str(log)})
    except Exception as exc:                     # 参考实现失败就原样返回，不阻断流程
        return gray, {"error": f"{type(exc).__name__}: {exc}"}


def deskew(gray: np.ndarray, min_angle: float = -5.0, max_angle: float = 5.0) -> Tuple[np.ndarray, float]:
    """熵最小化纠偏。返回 (纠正后图, 角度)。"""
    try:
        out, info = ref_deskew.deskew_image(gray, min_angle=min_angle, max_angle=max_angle)
        angle = float(info.get("skew_angle", 0.0)) if isinstance(info, dict) else 0.0
        return to_uint8_ink_dark(out), angle
    except Exception:
        return gray, 0.0


def run(gray: np.ndarray, do_relight: bool = True, do_deskew: bool = True,
        min_angle: float = -5.0, max_angle: float = 5.0) -> Tuple[np.ndarray, dict]:
    info: dict = {"relight": False, "deskew": False, "angle": 0.0}
    out = gray
    if do_relight:
        out, log = relight(out)
        info["relight"] = "error" not in log
        if "error" in log:
            info["relight_error"] = log["error"]
    if do_deskew:
        out2, ang = deskew(out, min_angle, max_angle)
        # 只在真的转了才用，避免无谓的插值损失
        if abs(ang) >= 0.05:
            out, info["deskew"], info["angle"] = out2, True, ang
    return out, info


def apply_to_bgr(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """把同样的纠偏角度应用到原彩图（保证重绘坐标与识别坐标一致）。"""
    if abs(angle_deg) < 0.05:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
