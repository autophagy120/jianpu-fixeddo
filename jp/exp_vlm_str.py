"""VLM 读谱行：只让模型吐「数字串」（不要坐标），三次采样投票 + 重叠拼接。"""
from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import urllib.request
from collections import Counter
from typing import List, Sequence, Tuple

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core

KEY = os.environ.get("DS_KEY", "")
BASE = "https://api.deepseek.com"

PROMPT = (
    "这是一条简谱谱行的一个片段。请把片段里**从左到右**的音符写成一串，用空格分开。\n"
    "写法：普通音符写数字 0-7；高八度（数字上方的点）在后面加 ^，两个点加 ^^；"
    "低八度（下方的点）加 _；升号写 #数字，降号写 b数字，都写在数字前面。\n"
    "只输出这一串音符，不要任何解释、不要标点、不要换行、不要代码块。\n"
    "示例输出：6 1^ 2 3# 5_ b7 0"
)


def ask(img, model="deepseek-flash", temperature=0.3, max_tokens=1200, timeout=180) -> List[str]:
    ok, enc = cv2.imencode(".png", img)
    b64 = base64.b64encode(enc.tobytes()).decode()
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    msg = body["choices"][0]["message"]
    txt = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    # 取最后一行像音符串的内容
    best: List[str] = []
    for line in txt.splitlines():
        toks = [t for t in line.replace(",", " ").split()
                if t and all(ch.isdigit() or ch in "^_#b" for ch in t)]
        if len(toks) > len(best) and len(toks) >= 3:
            best = toks
    return best


def vote(seqs: Sequence[List[str]]) -> List[str]:
    seqs = [s for s in seqs if s]
    if not seqs:
        return []
    n = Counter(len(s) for s in seqs).most_common(1)[0][0]
    same = [s for s in seqs if len(s) == n]
    out = []
    for i in range(n):
        out.append(Counter(s[i] for s in same).most_common(1)[0][0])
    return out


def merge_tiles(tiles: Sequence[Tuple[int, int, List[str]]], overlap_ratio: float = 0.22) -> List[str]:
    """tiles: [(x0, x1, tokens)] 按 x 排序；用重叠区的尾巴/头部匹配来拼接。"""
    if not tiles:
        return []
    out = list(tiles[0][2])
    for x0, x1, toks in tiles[1:]:
        if not toks:
            continue
        k = max(1, int(round(len(toks) * overlap_ratio)))
        best_shift, best_score = 0, -1
        for shift in range(1, min(k, len(toks), len(out)) + 1):
            if out[-shift:] == list(toks[:shift]):
                best_shift, best_score = shift, shift
        out.extend(toks[best_shift:] if best_shift else toks)
    return out


def read_line(bgr, y0, y1, tile_px=280, overlap_px=60, scale=5.0, samples=3, dbg=True):
    H, W = bgr.shape[:2]
    tiles = []
    x = 0
    while x < W:
        x2 = min(W, x + tile_px)
        crop = bgr[max(0, y0):min(H, y1), x:x2]
        if crop.shape[1] < 80:
            break
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        seqs = []
        for _ in range(samples):
            try:
                seqs.append(ask(big))
            except Exception as exc:
                seqs.append([])
        toks = vote(seqs)
        if dbg:
            print(f"    片 x={x}..{x2}: 采样长度 {[len(s) for s in seqs]} -> 投票 {len(toks)} 个 "
                  f"{' '.join(toks[:12])}", flush=True)
        tiles.append((x, x2, toks))
        if x2 >= W:
            break
        x = x2 - overlap_px
    return merge_tiles(tiles)


def main():
    P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
    bgr = core.load_bgr(P)
    print("=== 三月雨 第 1 行（真值 6123 循环）===")
    toks = read_line(bgr, 126, 172)
    print(f"\n共 {len(toks)} 个音符:\n  {' '.join(toks)}")
    gt = ["6", "1", "2", "3"] * 20
    ok = sum(1 for a, b in zip(toks, gt) if a.lstrip("#b").startswith(b))
    print(f"\n逐字正确 {ok}/{len(toks)}")


if __name__ == "__main__":
    main()
