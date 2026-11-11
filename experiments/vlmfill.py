"""用大模型补数字：小节线硬锚点分段对齐，避免"漏检导致整行平移"。

思路（针对"我漏检了怎么办"这个致命问题）：
1. 用我检测到的小节线把谱行切成若干**小节**；
2. 每一小节里，我的数字框数量 vs VLM 给的数字数量：
   - 相等 -> 按 x 顺序一一分配（高置信）
   - VLM 多   -> 只分配前 N 个，多出来的进"待核对"，**绝不插入**
   - VLM 少   -> 只能一对一分配前 N 个，剩下的保留我的 CNN 结果，该小节标"待核对"
3. 小节之间互不影响 —— 因此一处的数量差异不会平移整行。
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import core

KEY_ENV = ("DS_KEY", "DEEPSEEK_API_KEY")
BASE = "https://api.deepseek.com"

PROMPT = (
    "这是一条简谱谱行的图片，宽 __W__ 像素、高 __H__ 像素（尺寸已给出，不要推算）。\n"
    "逐音符输出：digit(0-7)、oct(1=上方一点,-1=下方一点,2/-2=两点,0=无点)、"
    "acc(\"\"或\"#\"或\"b\")、x(数字左边缘横坐标整数)、w(数字宽度整数)。\n"
    "只输出真实存在的音符；看不清就不要输出。\n"
    "要求：第一个字符必须是左花括号，只输出一个 JSON 对象，不要解释、不要思考过程、不要代码块。\n"
    '格式：{"notes":[{"digit":"6","oct":0,"acc":"","x":31,"w":12}],"line_width":__W__}'
)


def _key() -> str:
    for k in KEY_ENV:
        v = os.environ.get(k)
        if v:
            return v
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".deepseek_key")
    if os.path.exists(p):
        return open(p, encoding="utf8").read().strip()
    raise RuntimeError("缺少 DeepSeek key（环境变量 DS_KEY 或 .deepseek_key）")


def _ask(img_bgr, model="deepseek-flash", max_tokens=8000, timeout=300) -> str:
    h, w = img_bgr.shape[:2]
    ok, enc = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(enc.tobytes()).decode()
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.replace("__W__", str(w)).replace("__H__", str(h))},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + _key()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    msg = body["choices"][0]["message"]
    return (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()


def _parse(txt: str) -> Optional[dict]:
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


@dataclass
class FillReport:
    line_index: int
    bars: int = 0
    matched: int = 0
    extra_from_vlm: int = 0
    missing_from_vlm: int = 0
    flagged_bars: List[int] = field(default_factory=list)
    ok: bool = True
    note: str = ""


def vlm_digits_for_line(bgr: np.ndarray, page: core.Page, system: core.System,
                        upscale_to: int = 220, model: str = "deepseek-flash",
                        tile_w: int = 900, overlap: int = 70) -> List[dict]:
    """裁谱行 -> 按 ~900px 切片分别问 VLM -> 按 x 偏移拼起来。

    整行一条窄长图（长宽比 10:1 以上）会把推理模型的 token 全耗在思考上，
    切片后每片长宽比正常，能稳定吐出 JSON。
    """
    k = page.k or 1.0
    row = None
    for r in page.rows:
        if r.kind == "music" and abs(r.y0 - system.y0) < 2:
            row = r
            break
    top = row.y0 if row is not None else system.y0
    bot = row.y1 if row is not None else system.y0 + (system.y1 - system.y0) * 0.5
    pad = int(0.45 * page.scale)
    y0 = max(0, int(top / k) - pad)
    y1 = min(bgr.shape[0], int(bot / k) + pad)
    crop = bgr[y0:y1]
    if crop.shape[0] < 8:
        return []
    sc = max(1.0, upscale_to / float(crop.shape[0]))
    big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    H, W = big.shape[:2]

    out: List[dict] = []
    x = 0
    while x < W:
        x2 = min(W, x + tile_w)
        tile = big[:, x:x2]
        if tile.shape[1] < 120:                       # 太窄的尾片并到上一片
            if out:
                break
        txt = _ask(tile, model=model)
        data = _parse(txt)
        if data is None and os.environ.get("JP_DEBUG_VLM"):
            print(f"    [VLM 解析失败] tile x={x}..{x2} 原文: {txt[:120]!r}", flush=True)
        if data:
            for n in data.get("notes", []):
                try:
                    nx = x + float(n.get("x", -1))
                    out.append({"digit": str(n.get("digit")), "oct": int(n.get("oct", 0) or 0),
                                "acc": (n.get("acc") or ""), "x": nx / sc,
                                "w": float(n.get("w", 0)) / sc})
                except Exception:
                    continue
        if x2 >= W:
            break
        x = x2 - overlap

    out.sort(key=lambda d: d["x"])
    # 去掉重叠区里重复的音符（x 接近且 digit 相同）
    dedup: List[dict] = []
    for d in out:
        if dedup and abs(d["x"] - dedup[-1]["x"]) < 0.45 * max(1.0, d.get("w", 1)):
            continue
        dedup.append(d)
    return dedup


def fill_line(page: core.Page, system: core.System, vlm_notes: Sequence[dict],
              bar_xs: Sequence[float]) -> FillReport:
    """把 VLM 的数字按小节对齐到我的音符上。原地修改 system.notes。"""
    rep = FillReport(line_index=system.row_index)
    notes = [n for n in system.notes if n.digit != "0"]
    if not notes or not vlm_notes:
        rep.ok = False
        rep.note = "无音符/无 VLM 结果"
        return rep
    k = page.k or 1.0
    edges = [0.0] + [float(x) / k for x in bar_xs] + [10 ** 9]
    rep.bars = len(edges) - 1
    for bi in range(len(edges) - 1):
        lo, hi = edges[bi], edges[bi + 1]
        mine = [n for n in notes if lo <= n.x < hi]
        theirs = [d for d in vlm_notes if lo <= d["x"] < hi]
        if not mine or not theirs:
            if mine or theirs:
                rep.flagged_bars.append(bi)
                rep.missing_from_vlm += len(mine)
            continue
        n_use = min(len(mine), len(theirs))
        for i in range(n_use):
            n, d = mine[i], theirs[i]
            if d["digit"] in "01234567":
                n.digit = d["digit"]
                n.conf = max(n.conf, 0.75)
                if d["oct"]:
                    n.octave = d["oct"]
                if d["acc"] in ("#", "b"):
                    n.acc = d["acc"]
                rep.matched += 1
        if len(theirs) > len(mine):
            rep.extra_from_vlm += len(theirs) - len(mine)
            rep.flagged_bars.append(bi)
        elif len(mine) > len(theirs):
            rep.missing_from_vlm += len(mine) - len(theirs)
            rep.flagged_bars.append(bi)
    return rep
