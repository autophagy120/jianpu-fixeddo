"""给语料级模板库命名：每 6 个簇一组交给 VLM，问两次取一致，结果写回 npz。"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from . import core
from . import pagetempl as PT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BANK_PATH = os.path.join(ROOT, "models", "templ_bank.npz")

PROMPT = (
    "图中每一格是一个**印刷体数字字形**（来自同一批中文简谱），格子左上角标着它的编号。\n"
    "请判断每格是哪个数字：0 1 2 3 4 5 6 7；"
    "如果它不是完整的数字（比如只有半个字、横线、黑块、括号、噪点），写 other。\n"
    "不要逐条分析，直接给结论。只输出 JSON，第一个字符必须是左花括号：\n"
    '{"0":"6","1":"other", ...}  键=编号，值=数字或 other。'
)


def montage(masks: Sequence[np.ndarray], labels_idx: Sequence[int], tile: int = 110,
            cols: int = 3) -> np.ndarray:
    rows = int(np.ceil(len(masks) / cols))
    canvas = np.full((rows * (tile + 42), cols * (tile + 14), 3), 255, np.uint8)
    for k, (m, gi) in enumerate(zip(masks, labels_idx)):
        r, c = divmod(k, cols)
        y0, x0 = r * (tile + 42) + 32, c * (tile + 14) + 7
        big = cv2.resize(255 - m * 255, (tile, tile), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + tile + 1, y0 + tile + 1), (160, 160, 160), 1)
        cv2.putText(canvas, f"#{k}", (x0, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 210), 2)
    return cv2.resize(canvas, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def ask(img: np.ndarray, retries: int = 3) -> Dict[str, str]:
    for _ in range(retries):
        got = PT.ask_labels(img)          # 复用同一套 JSON 解析
        if got:
            return got
    return {}


def label_bank(top_n: int = 120, chunk: int = 6) -> Tuple[np.ndarray, List[str], np.ndarray]:
    z = np.load(BANK_PATH, allow_pickle=True)
    cs, labels, counts = z["centroids"], [str(x) for x in z["labels"]], z["counts"]
    order = list(np.argsort(-counts)[:top_n])
    print(f"给前 {len(order)} 个簇命名（覆盖 {counts[order].sum() / counts.sum():.1%}）", flush=True)

    named: Dict[int, str] = {}
    for start in range(0, len(order), chunk):
        part = order[start:start + chunk]
        img = montage([cs[i] for i in part], part, cols=min(3, len(part)))
        a = ask(img)
        b = ask(img)                       # 第二次，取一致
        agree = {}
        for k, gi in enumerate(part):
            la, lb = a.get(str(k), "?"), b.get(str(k), "?")
            if la == lb and la != "?":
                agree[gi] = la
            elif la != "?" and lb == "?":
                agree[gi] = la
            elif la == "?" and lb != "?":
                agree[gi] = lb
            else:
                agree[gi] = la if la == lb else "?"
        named.update(agree)
        print(f"  组 {start//chunk + 1}: 编号{list(part)} -> {[agree.get(i,'?') for i in part]}", flush=True)

    for i in range(len(labels)):
        if i in named:
            labels[i] = named[i]
    np.savez_compressed(BANK_PATH, centroids=cs, labels=np.array(labels), counts=counts)
    ok = sum(1 for i in range(len(labels)) if labels[i] in "01234567")
    cov = sum(int(counts[i]) for i in range(len(labels)) if labels[i] in "01234567") / counts.sum()
    print(f"\n命名完成：{ok} 簇有数字标签，覆盖 {cov:.1%} 的字形")
    return cs, labels, counts


def review_sheet(cs, labels, counts, out_path: str, top_n: int = 120):
    order = list(np.argsort(-counts)[:top_n])
    tile, cols = 110, 10
    rows = int(np.ceil(len(order) / cols))
    canvas = np.full((rows * (tile + 42), cols * (tile + 12), 3), 255, np.uint8)
    for k, idx in enumerate(order):
        r, c = divmod(k, cols)
        y0, x0 = r * (tile + 42) + 32, c * (tile + 12) + 6
        big = cv2.resize(255 - cs[idx] * 255, (tile, tile), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + tile, y0 + tile), (185, 185, 185), 1)
        lab = labels[idx]
        col = (30, 30, 200) if lab in "01234567" else (150, 150, 150)
        cv2.putText(canvas, f"{lab}", (x0 + 38, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    core.imwrite(out_path, canvas)
    print(f"复核图 {out_path}")


if __name__ == "__main__":
    cs, labels, counts = label_bank()
    review_sheet(cs, labels, counts, os.path.join(ROOT, "out", "bank_labeled.png"))
