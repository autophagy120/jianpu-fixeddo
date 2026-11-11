"""页内自建模板：把这一页的字形聚类，只让 VLM 认"每类代表"，再把标签铺回整页。

动机：差分辨率页（9 px 的字）逐字分类一定不稳，但同一字体在整页里重复几十次。
先聚类（原生尺寸、不缩放），每类挑代表拼成一张图交给 VLM 认一次，
再用"到类中心的距离"给每个字形贴标签。一次 VLM 调用 = 整页分类表。
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

BASE = "https://api.deepseek.com"


# ------------------------------------------------------------------ 聚类
def clean_patch(patch: np.ndarray) -> np.ndarray:
    """剥掉贴在数字上的细横线（减时线残余）与孤立小点，再做后续比对。

    9 px 的字，减时线常常和数字粘在一起；同一个 `1` 会因此分成"光洁版/带线版"两类，
    聚类和比对都会乱。
    """
    a = patch if patch.dtype == np.uint8 else patch.astype(np.uint8)
    if a.size == 0:
        return a
    thin_h = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    thin_v = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    out = cv2.bitwise_and(a, cv2.bitwise_not(cv2.bitwise_or(thin_h, thin_v)))
    # 去掉极小孤立点
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    for i in range(1, n):
        if stats[i, 4] < 4:
            out[lab == i] = 0
    ys, xs = np.nonzero(out)
    if len(ys) == 0:
        return a
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _norm_mask(patch: np.ndarray, size: int = 24) -> np.ndarray:
    """原生尺寸的小块 -> 统一画布（保持宽高比，短边居中），用于相似度比较。"""
    a = patch if patch.dtype == np.uint8 else patch.astype(np.uint8)
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return np.zeros((size, size), np.uint8)
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    s = (size - 4) / float(max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    res = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_NEAREST)
    can = np.zeros((size, size), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    can[oy:oy + nh, ox:ox + nw] = res
    return (can > 100).astype(np.uint8)


@dataclass
class Cluster:
    members: List[int] = field(default_factory=list)   # 索引
    label: str = "?"
    sim: float = 0.0


def cluster_patches(patches: Sequence[np.ndarray], sim: float = 0.55) -> List[Cluster]:
    masks = [_norm_mask(clean_patch(p)) for p in patches]
    clusters: List[Cluster] = []
    for i, m in enumerate(masks):
        best, best_s = None, -1.0
        for c in clusters:
            cm = masks[c.members[0]]
            inter = np.logical_and(m, cm).sum()
            union = np.logical_or(m, cm).sum()
            s = inter / union if union else 0.0
            if s > best_s:
                best, best_s = c, s
        if best is not None and best_s >= sim:
            best.members.append(i)
        else:
            clusters.append(Cluster(members=[i]))
    for c in clusters:
        c.sim = float(len(c.members))
    clusters.sort(key=lambda c: -len(c.members))
    # 只保留够大的类，零碎的统一丢进 other
    big = [c for c in clusters if len(c.members) >= 4][:10]
    small = [c for c in clusters if c not in big]
    if small:
        merged = Cluster(members=[m for c in small for m in c.members], label="other")
        big.append(merged)
    return big


# ------------------------------------------------------------------ 拼图 + VLM
def _montage(patches: Sequence[np.ndarray], clusters: Sequence[Cluster], tile: int = 96,
             cols: int = 6, scale: int = 3) -> Tuple[np.ndarray, List[int]]:
    """每个类一张代表图，拼成带序号的图，放大后交给 VLM。"""
    reps = []
    for c in clusters:
        reps.append(patches[c.members[0]])
    rows = int(np.ceil(len(reps) / cols))
    canvas = np.full((rows * (tile + 34), cols * (tile + 12), 3), 255, np.uint8)
    idxs = []
    for i, p in enumerate(reps):
        r, cc = divmod(i, cols)
        y0 = r * (tile + 34) + 26
        x0 = cc * (tile + 12) + 6
        a = p if p.dtype == np.uint8 else p.astype(np.uint8)
        big = cv2.resize(a, (tile, tile), interpolation=cv2.INTER_NEAREST)
        big = 255 - big                       # 白底黑字
        canvas[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + tile + 1, y0 + tile + 1), (200, 200, 200), 1)
        cv2.putText(canvas, f"#{i}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1)
        idxs.append(i)
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return canvas, idxs


PROMPT = (
    "这张图里每个灰框内是一个**简谱数字字形**（印刷体，来自同一本谱），框上方标着它的编号如 #0。\n"
    "请判断每个编号对应哪个数字，可能的取值：0 1 2 3 4 5 6 7，"
    "或者 other（不是数字，比如括号、小节线、连音线片段、歌词残片）。\n"
    "要求：不要逐个分析、不要写推理过程，**直接给出最终答案**。\n"
    "只输出 JSON，第一个字符必须是左花括号，不要解释：\n"
    '{"0":"6","1":"1","2":"other", ...}  键是编号数字，值是上面的取值。'
)


def _key() -> str:
    for k in ("DS_KEY", "DEEPSEEK_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".deepseek_key")
    if os.path.exists(p):
        return open(p, encoding="utf8").read().strip()
    raise RuntimeError("缺少 DeepSeek key")


def ask_labels(montage: np.ndarray, model: str = "deepseek-flash", max_tokens: int = 9000) -> Dict[str, str]:
    ok, enc = cv2.imencode(".png", montage)
    b64 = base64.b64encode(enc.tobytes()).decode()
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + _key()})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read().decode())
    msg = body["choices"][0]["message"]
    txt = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    if os.environ.get("JP_DEBUG_VLM"):
        print(f"    [VLM 原始] {txt[:260]!r}", flush=True)
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        return {}
    try:
        raw = json.loads(txt[i:j + 1])
    except Exception:
        return {}
    return {str(k).lstrip("#"): str(v).strip() for k, v in raw.items()}


def assign_by_centroids(patches, clusters, clusters_with_label, min_iou: float = 0.52,
                        margin: float = 0.06):
    """把每个字形分配给"已用 VLM 标好标签"的类中心里最像的那个。

    比"聚类时的首次匹配"稳：聚类只用来产出候选中心，最终归属看全局最优。
    返回 (labels, confs)；不够确信的给 None（进待核对）。
    """
    masks = [_norm_mask(clean_patch(p)) for p in patches]
    cents = []          # (label, mask)
    for ci in clusters_with_label:
        c = clusters[ci]
        centre = masks[c.members[0]]
        cents.append((c.label, centre))
    labels, confs = [], []
    if not cents:
        return [None] * len(masks), [0.0] * len(masks)
    for m in masks:
        scored = []
        for lab, cm in cents:
            inter = np.logical_and(m, cm).sum()
            union = np.logical_or(m, cm).sum()
            scored.append((inter / union if union else 0.0, lab))
        scored.sort(reverse=True)
        best_s, best_l = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_l in ("other", "?") or best_s < min_iou:
            labels.append(None)
            confs.append(best_s)
        elif best_s - second < margin:
            labels.append(None)
            confs.append(best_s)
        else:
            labels.append(best_l)
            confs.append(best_s)
    return labels, confs


def label_clusters(patches, clusters, chunk: int = 6, retries: int = 3, debug: bool = False):
    """把类别分成小块分别交给 VLM 认（小块更稳），失败的块重试。"""
    labels = {}
    idxs = list(range(len(clusters)))
    for start in range(0, len(idxs), chunk):
        part = idxs[start:start + chunk]
        if not part:
            continue
        sub = [clusters[i] for i in part]
        montage, _ = _montage(patches, sub, cols=min(3, len(sub)))
        got = {}
        for attempt in range(retries):
            got = ask_labels(montage)
            if got:
                break
        # 子图里的编号是 0..len(part)-1，要映射回全局编号
        for k, gi in enumerate(part):
            labels[gi] = got.get(str(k), "?")
        if debug:
            print(f"    块 {start}-{start + len(part) - 1}: {got}", flush=True)
    return labels
