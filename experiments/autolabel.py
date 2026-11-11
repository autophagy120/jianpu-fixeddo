"""用 easytr 成品图自动标注模板库（不依赖任何 API）。

原理：
1. easytr 的成品图与原图**版式完全相同**，只是把数字重绘了（用 msyh 字体）；
2. 既然重绘字体已知（msyh），我可以在**原尺寸**上用 msyh 模板匹配读出成品图每个数字
   —— 字体、字号都对得上，这是确定性高准确率的；
3. 再用「转调逆映射」把成品图上的数字还原成原谱数字；
4. 原图同一坐标上的字形（我聚类后的 mask）就得到了标签 → 投票得出每个簇的数字。
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core
from . import music as M
from . import templbank as TB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MSYH = r"C:\Windows\Fonts\msyh.ttc"
PRODUCED = re.compile(r"\(原调\s*([#b]?[A-Ga-g])\s*-\s*转调\s*([#b]?[A-Ga-g])\)")


# ---------------------------------------------------------------- 模板
def msyh_digits(height: int) -> Dict[str, np.ndarray]:
    """渲染高度≈height 的 msyh 数字（原生尺寸，不缩放）。"""
    out: Dict[str, np.ndarray] = {}
    for ch in "01234567":
        best = None
        for fs in range(max(5, height - 2), height + 14):
            try:
                f = ImageFont.truetype(MSYH, fs)
            except Exception:
                break
            img = Image.new("L", (fs * 3, fs * 3), 0)
            ImageDraw.Draw(img).text((fs, fs), ch, fill=255, font=f)
            a = np.array(img)
            ink = (a > 110).astype(np.uint8)
            ys, xs = np.nonzero(ink)
            if len(ys) == 0:
                continue
            sub = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if best is None or abs(sub.shape[0] - height) < abs(best.shape[0] - height):
                best = sub
            if abs(sub.shape[0] - height) <= 0:
                break
        if best is not None:
            out[ch] = best
    return out


def match_digit(patch: np.ndarray, bank: Dict[str, np.ndarray], min_iou: float = 0.60,
                margin: float = 0.05) -> Optional[str]:
    a = patch > 0
    if a.sum() == 0:
        return None
    scored = []
    for ch, t in bank.items():
        if abs(t.shape[0] - a.shape[0]) > 2 or abs(t.shape[1] - a.shape[1]) > 3:
            continue
        bb = t > 0
        best = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                b2 = np.roll(np.roll(bb, dy, axis=0), dx, axis=1)
                h, w = min(a.shape[0], b2.shape[0]), min(a.shape[1], b2.shape[1])
                if h <= 0 or w <= 0:
                    continue
                aa, cc = a[:h, :w], b2[:h, :w]
                u = np.logical_or(aa, cc).sum()
                if u:
                    best = max(best, float(np.logical_and(aa, cc).sum()) / u)
        scored.append((best, ch))
    if not scored:
        return None
    scored.sort(reverse=True)
    if scored[0][0] < min_iou:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][1]


# ---------------------------------------------------------------- 逆映射
def inverse_map(key_from: int, key_to: int) -> Dict[str, Tuple[str, int]]:
    """成品图上的数字(首调写法) -> 原谱的 (数字, 八度点变化)。

    转调是"绝对音高不变、写法改变"：原谱 digit+acc 在 key_from 下对应一个绝对音高，
    成品图用 key_to 重写。逆映射只需枚举 1..7 与 acc，看它转过去变成什么。
    """
    inv: Dict[str, Tuple[str, int]] = {}
    for d in range(1, 8):
        for acc in ("", "#", "b"):
            try:
                p = M.note_to_abs(d, acc, 0, key_from)
                nd, nacc, noct = M.abs_to_movable(p, key_to)
            except Exception:
                continue
            inv[f"{nacc}{nd}{noct}"] = (f"{acc}{d}", 0)
    return inv


# ---------------------------------------------------------------- 读成品图
def produced_digits(path: str) -> Dict[Tuple[int, int], str]:
    """成品图上每个数字：位置(量化到 4px 网格) -> 数字。"""
    try:
        g = core.load_gray(path)
    except Exception:
        return {}
    bw = core.binarize(g)
    comps = core.components(bw, min_area=4)
    hs = [c.h for c in comps if c.h >= 5 and 0.3 <= c.aspect <= 1.0]
    if not hs:
        return {}
    h_med = int(np.median(hs))
    bank = msyh_digits(h_med)
    out: Dict[Tuple[int, int], str] = {}
    for c in comps:
        if abs(c.h - h_med) > max(2, 0.35 * h_med):
            continue
        if not (0.2 <= c.aspect <= 1.1):
            continue
        patch = bw[c.y:c.y + c.h, c.x:c.x + c.w]
        d = match_digit(patch, bank)
        if d is not None:
            out[(c.x // 4, c.y // 4)] = d
    return out


def original_glyphs(path: str) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    """原图上"像数字"的字形：位置(量化到 4px) -> 清洗后的 mask。"""
    try:
        g = core.load_gray(path)
    except Exception:
        return []
    bw = core.binarize(g)
    comps = core.components(bw, min_area=4)
    hs = [c.h for c in comps if c.h >= 6 and 0.3 <= c.aspect <= 1.0]
    if len(hs) < 15:
        return []
    h_med = float(np.median(hs))
    out = []
    for c in comps:
        if abs(c.h - h_med) > max(2, 0.4 * h_med):
            continue
        if not (0.18 <= c.aspect <= 1.1):
            continue
        patch = bw[c.y:c.y + c.h, c.x:c.x + c.w]
        if patch.size == 0:
            continue
        out.append(((c.x // 4, c.y // 4), TB.norm_mask(TB.clean(patch))))
    return out


# ---------------------------------------------------------------- 汇总
def find_pairs(corpus: str = TB.CORPUS) -> List[Tuple[str, str, int, int]]:
    pairs = []
    for root, _d, fs in os.walk(corpus):
        produced = {}
        originals = {}
        for f in sorted(fs):
            m = PRODUCED.search(f)
            if m:
                stem = os.path.splitext(PRODUCED.sub("", f))[0].strip()
                produced.setdefault(stem, (f, m.group(1), m.group(2)))
            elif os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"):
                originals[os.path.splitext(f)[0]] = f
        for stem, (pf, k1, k2) in produced.items():
            if stem not in originals:
                continue
            kf = M.parse_key(k1)
            kt = M.parse_key(k2)
            if kf is None or kt is None:
                continue
            pairs.append((os.path.join(root, originals[stem]), os.path.join(root, pf), kf, kt))
    return pairs


def main():
    corpus = TB.CORPUS
    pairs = find_pairs(corpus)
    print(f"{len(pairs)} 组(原图, 成品图)")

    # 载入已有簇中心，用于投票
    z = np.load(TB.BANK_PATH, allow_pickle=True)
    cs, labels, counts = z["centroids"], [str(x) for x in z["labels"]], z["counts"]

    votes: List[Counter] = [Counter() for _ in range(len(cs))]
    total = 0
    for i, (orig, prod, kf, kt) in enumerate(pairs, 1):
        pd = produced_digits(prod)
        gl = original_glyphs(orig)
        if not pd or not gl:
            print(f"  [{i}/{len(pairs)}] 跳过（成品读数 {len(pd)} / 原图字形 {len(gl)}）", flush=True)
            continue
        inv = inverse_map(kf, kt)
        used = 0
        for pos, mask in gl:
            # 就近找成品图同一位置的数字
            cands = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    hit = pd.get((pos[0] + dx, pos[1] + dy))
                    if hit:
                        cands.append(hit)
            if not cands:
                continue
            d_prod = Counter(cands).most_common(1)[0][0]
            # 逆映射：成品写法 -> 原谱写法
            back = None
            for k, v in inv.items():
                if k.startswith(d_prod) or k == d_prod:
                    back = v
                    break
            if back is None:
                continue
            label = back[0].lstrip("#b")
            if label not in "01234567":
                continue
            # 找最近的簇
            best_i, best_s = -1, -1.0
            for ci, c in enumerate(cs):
                u = np.logical_or(mask > 0, c > 0).sum()
                s = float(np.logical_and(mask > 0, c > 0).sum()) / u if u else 0.0
                if s > best_s:
                    best_i, best_s = ci, s
            if best_i >= 0 and best_s >= 0.70:
                votes[best_i][label] += 1
                used += 1
        total += used
        print(f"  [{i}/{len(pairs)}] {os.path.basename(orig)[:34]:34s} 投票 {used}", flush=True)

    print(f"\n共为 {total} 个字形建立 [字形->数字] 对应")
    changed = 0
    for i, v in enumerate(votes):
        if not v:
            continue
        lab, n = v.most_common(1)[0]
        share = n / sum(v.values())
        if share >= 0.6 and n >= 15:
            if labels[i] != lab:
                labels[i] = lab
                changed += 1
    np.savez_compressed(TB.BANK_PATH, centroids=cs, labels=np.array(labels), counts=counts)
    named = [(i, labels[i], int(counts[i])) for i in range(len(labels)) if labels[i] in "01234567"]
    cov = sum(c for _i, _l, c in named) / counts.sum()
    print(f"写回模板库：命名 {len(named)} 簇（改动 {changed}），覆盖 {cov:.1%} 的字形")
    for lab in "01234567":
        got = [(i, c) for i, l, c in named if l == lab]
        print(f"  数字 {lab}: {len(got)} 簇，共 {sum(c for _i, c in got)} 例")


if __name__ == "__main__":
    main()
