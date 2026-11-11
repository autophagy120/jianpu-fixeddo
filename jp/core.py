"""Image -> systems -> notes.  The recognition half of the pipeline."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


# ------------------------------------------------------------------ io
def load_gray(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_bgr(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(path)
    buf.tofile(path)


# ------------------------------------------------------------- preprocess
def binarize(gray: np.ndarray) -> np.ndarray:
    # light denoise first: jpeg ringing creates specks that survive Otsu
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if float((bw > 0).mean()) > 0.5:          # inverted scan
        bw = 255 - bw
    return bw


def strip_rules(bw: np.ndarray, h_len: int, v_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """去掉**细**长横线（减时线 / 延音线）。

    关键：只看"细"的横线。粗体数字自己的横笔（3-5 px 粗）跟减时线（1-2 px 粗）
    长度相近，只用长度开运算会把数字切成左右两半——上一版就是这么把一半音符
    判成 'other' 的。所以先用 (1,3) 竖向腐蚀筛掉"厚的"，再按长度取横线。

    竖线（小节线）一律不动：1-2 px 的小节线和数字 1 的竖笔一样宽，任何形态学
    操作都会吃掉音符；竖线改为按几何形状过滤。
    """
    thin = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    hor_thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    hor = cv2.dilate(hor_thin, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    clean = cv2.bitwise_and(bw, cv2.bitwise_not(hor))
    return clean, cv2.bitwise_or(hor, ver)


@dataclass
class Comp:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def box(self) -> Box:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def aspect(self) -> float:
        return self.w / max(1.0, float(self.h))


def components(bw: np.ndarray, min_area: int = 4) -> List[Comp]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        out.append(Comp(int(x), int(y), int(w), int(h), int(a)))
    return out


def is_rule_like(c: Comp, scale: float) -> bool:
    """Barline / repeat-sign / underline leftovers that must never become a note."""
    if c.h > 1.35 * scale and c.w < 0.22 * scale:          # vertical bar
        return True
    if c.w > 2.4 * scale and c.h < 0.35 * scale:           # horizontal rule fragment
        return True
    if c.w < 0.1 * scale and c.h < 0.5 * scale:
        return True
    return False


def patch_of(bw: np.ndarray, c: Comp, pad: int = 1) -> np.ndarray:
    h, w = bw.shape
    x0, y0 = max(0, c.x - pad), max(0, c.y - pad)
    x1, y1 = min(w, c.x + c.w + pad), min(h, c.y + c.h + pad)
    return bw[y0:y1, x0:x1]


# ----------------------------------------------------------------- rows
@dataclass
class Row:
    y0: int
    y1: int
    comps: List[Comp]
    kind: str = "unknown"      # music | lyric | header | other
    digit_frac: float = 0.0
    conf_mean: float = 0.0


def row_bands(clean: np.ndarray, min_h: int = 4) -> List[Tuple[int, int]]:
    prof = (clean > 0).sum(axis=1).astype(np.float64)
    xs = np.nonzero(prof > 0)[0]
    if len(xs) == 0:
        return []
    bands: List[List[int]] = []
    s = xs[0]
    prev = xs[0]
    for y in xs[1:]:
        if y - prev > 2:
            bands.append([s, prev + 1])
            s = y
        prev = y
    bands.append([s, prev + 1])
    # merge slivers created by dot rows into their neighbour when very close
    merged: List[List[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= 1:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(a, b) for a, b in merged if b - a >= min_h]


def glyph_scale(comps: Sequence[Comp]) -> float:
    """Dominant *glyph* height (digits/lyrics), robust to specks and rule fragments.

    只用有肉的字形：面积够大、宽高比像数字（0.3-1.0）。旧版取直方图众数，
    会在残留碎屑很多时被 6-8 px 的小块带偏（重绘后的图尤其明显）。
    """
    cand = [c for c in comps if c.h >= 8 and c.area >= 25 and 0.30 <= c.aspect <= 1.0]
    if len(cand) < 8:
        cand = [c for c in comps if c.h >= 6 and c.area >= 12]
    if not cand:
        return 12.0
    hs = np.array([c.h for c in cand], dtype=np.float64)
    return float(np.median(hs))


@dataclass
class Note:
    digit: str
    acc: str = ""            # '' | '#' | 'b'
    octave: int = 0          # >=0 up dots, <0 down dots
    x: float = 0.0
    box: Box = (0, 0, 0, 0)
    acc_box: Optional[Box] = None
    dot_boxes: List[Box] = field(default_factory=list)
    dot_after: List[Box] = field(default_factory=list)
    conf: float = 1.0
    row: int = 0
    # 字号类别：`main` 主字号（按整页中位字高）/ `aux` 辅助字号（和声行、装饰音等，
    # 按 `aux_ratio` 缩小）。见 ROI工具需求 C。
    size_class: str = "main"


@dataclass
class System:
    row_index: int
    y0: int
    y1: int
    notes: List[Note]
    lyric_rows: List[Tuple[int, int]]
    x0: int = 0
    x1: int = 0
    bar_boxes: List[Box] = field(default_factory=list)


@dataclass
class Page:
    path: str
    gray: np.ndarray
    clean: np.ndarray
    scale: float
    rows: List[Row]
    systems: List[System]
    comps: List[Comp] = field(default_factory=list)
    rules: Optional[np.ndarray] = None
    dash: Optional[np.ndarray] = None   # 短横线（减时线残余）掩码，用来否掉'假八度点'
    warnings: List[str] = field(default_factory=list)
    k: float = 1.0

    def comps_in(self, y0: int, y1: int) -> List[Comp]:
        return [c for c in self.comps if y0 - 1 <= c.cy < y1 + 1]


# --------------------------------------------------------------- pipeline
def analyse(gray: np.ndarray, net, path: str = "", debug: Dict = None,
            digit_clf=None) -> Page:
    """Full page analysis. `net` is a GlyphNet (or None -> geometry only)."""
    h, w = gray.shape
    bw0 = binarize(gray)
    comps0 = components(bw0, min_area=3)
    scale = glyph_scale(comps0)
    # 低分辨率页（字高 < 18px）先做 SRCNN 3× 超分 + 锐化，再走原有的缩放/二值化。
    # 实测（同几何、同 OCR 文本，`jp.preproc_cmp`）：
    #   三月雨（9px）raw 80.5% → sr3+sharp 85.9%（+5.4 点）
    #   为了你（高清）raw 98.8% 反而是最优，超分后掉到 96.2%
    # 所以只对低分辨率页启用，高清页保持原路径。
    if scale and scale < 18:
        try:
            from . import sr as _sr
            if _sr.available():
                _bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gray = cv2.cvtColor(_sr.upscale(_bgr, 3), cv2.COLOR_BGR2GRAY)
                _b = cv2.GaussianBlur(gray, (0, 0), 1.2)
                gray = cv2.addWeighted(gray, 1.8, _b, -0.8, 0)
                bw0 = binarize(gray)
                comps0 = components(bw0, min_area=3)
                scale = glyph_scale(comps0)
        except Exception:
            pass
    # normalise working resolution.  低分辨率扫描（数字 <18 px）用更大的工作尺寸，
    # 否则二值化会把 1-2 px 的笔画撕开，减时线又会和数字粘成一个块。
    if scale and scale < 18:
        target = 38.0
    elif scale and scale < 24:
        target = 32.0
    else:
        target = 28.0
    k = 1.0
    if scale > 0:
        k = float(np.clip(target / scale, 0.5, 6.0))
    if abs(k - 1.0) > 0.06:
        gray_w = cv2.resize(gray, (int(round(w * k)), int(round(h * k))),
                            interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        if k > 1.5:
            gray_w = cv2.bilateralFilter(gray_w, 3, 25, 25)
            blur = cv2.GaussianBlur(gray_w, (0, 0), 1.2)
            gray_w = cv2.addWeighted(gray_w, 1.55, blur, -0.55, 0)
    else:
        gray_w = gray
    bw = binarize(gray_w)
    comps = components(bw, min_area=4)
    scale_w = glyph_scale(comps)
    clean, rules = strip_rules(bw, max(9, int(0.75 * scale_w)), max(6, int(0.55 * scale_w)))
    # 短横线：比正常减时线短、但比八度点长得多，单独开运算取出来
    dash = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(0.42 * scale_w)), 1)))
    comps = components(clean, min_area=4)
    scale_w = glyph_scale(comps)

    dash_local = dash
    bands = row_bands(clean, min_h=max(4, int(0.22 * scale_w)))
    rows: List[Row] = []
    for y0, y1 in bands:
        cs = [c for c in comps if c.cy >= y0 - 1 and c.cy < y1 + 1 and not is_rule_like(c, scale_w)]
        if not cs:
            continue
        rows.append(Row(y0, y1, cs))

    # --- classify every glyph-sized component once, then label the rows
    # 用页级字高做门槛（方案 B 的"行内局部字高"已回退：它会改变行分类，
    # 实测把整行判成歌词行、造成整行丢失，三月雨配对 441→417）。
    glyph_min_h = 0.42 * scale_w
    row_scale: Dict[int, float] = {}
    idx: List[int] = []
    patches: List[np.ndarray] = []
    for r, row in enumerate(rows):
        gh_min = glyph_min_h
        for c in row.comps:
            if c.h >= gh_min and c.area >= 8:
                idx.append((r, id(c)))
                patches.append(patch_of(clean, c))
    labels: List[Optional[str]] = [None] * len(patches)
    confs = np.zeros(len(patches), dtype=np.float32)
    if net is not None and patches:
        labels, confs = net.digit_with_reject(patches)
        # second pass: acc class for rejected-but-small marks is handled per-note later
    lab_map = {}
    for i, key in enumerate(idx):
        lab_map[key[1]] = (labels[i], float(confs[i]))

    for r, row in enumerate(rows):
        tot = 0
        dig = 0
        cs = 0.0
        gh_min = glyph_min_h
        for c in row.comps:
            if c.h < gh_min or c.area < 8:
                continue
            lab, cf = lab_map.get(id(c), (None, 0.0))
            tot += 1
            if lab is not None:
                dig += 1
            cs += cf
        row.digit_frac = dig / tot if tot else 0.0
        row.conf_mean = cs / tot if tot else 0.0
        xspan = (max(c.x + c.w for c in row.comps) - min(c.x for c in row.comps)) if row.comps else 0
        # `xspan` 去掉大部分限制：`6 6 5 -`、`(3) 3· 4 4 1` 这类短行
        # 本来只占页宽一小段，用 30% 页宽当门槛会把它们整行判掉。
        if tot >= 3 and row.digit_frac >= 0.22 and xspan > 0.08 * clean.shape[1]:
            row.kind = "music"
        elif tot >= 3 and row.digit_frac < 0.28:
            row.kind = "lyric"
        else:
            row.kind = "other"

    warnings: List[str] = []

    # --- 行类型几何兜底
    # 谱行特征：字形高度接近全页最大、字间距均匀、行内竖线（小节线）多。
    # CNN 只是加速器，不能让"预处理改了笔画"就把整页判成歌词。
    def geometric_music_rows(rows_, scale_):
        """几何判谱行：谱行里的数字**高度几乎一致**、间距均匀；歌词行高矮不齐。"""
        out_ = []
        for r_ in rows_:
            glyphs = sorted([c for c in r_.comps if c.h >= 0.45 * scale_], key=lambda c: c.x)
            if len(glyphs) < 3:
                continue
            hs = np.array([g.h for g in glyphs], dtype=np.float64)
            cv = float(hs.std() / (hs.mean() + 1e-6))          # 高度离散度
            if cv > 0.18:
                continue
            gaps = np.diff([g.cx for g in glyphs])
            if len(gaps) == 0:
                continue
            regularity = 1.0 - float(gaps.std() / (gaps.mean() + 1e-6))
            if regularity >= 0.45 and cv <= 0.18:
                out_.append(r_)
        return out_

    # 几何兜底**总是执行**（原来只在"一个谱行都没找到"时跑）。
    # 小字号和声行（`(3) 3· 4 4 1 |`）常被判成歌词行，于是整行不扫描；
    # 而几何判据看的是"字高一致 + 间距均匀"，中文歌词高矮不齐通不过，
    # 小字音符行高度整齐、能通过 —— 所以追加标记是安全的。
    _geo_rows = geometric_music_rows(rows, scale_w)
    _added = 0
    for r in _geo_rows:
        if r.kind != "music":
            r.kind = "music"
            _added += 1
    if _added:
        warnings.append(f"几何追加 {_added} 条谱行（含小字号行）")

    # --- assemble systems: a music row + following lyric rows (and dot strips)
    # 全页音乐行的字形：一次性分类 + 页内自洽，结果按 id(comp) 存起来给 build_notes 用
    music_comps = []
    for r in rows:
        if r.kind != "music":
            continue
        for c0 in r.comps:
            for c in split_wide_component(clean, c0, scale_w):
                if c.h >= 0.45 * scale_w and c.area >= 8 and (0.14 <= c.aspect <= 1.35):
                    music_comps.append(c)
    page_labels: Dict[int, tuple] = {}
    if music_comps and (digit_clf is not None or net is not None):
        mp = [patch_of(clean, c) for c in music_comps]
        if digit_clf is not None:
            # 原生尺寸模板库：字体匹配时几乎像素级重合，低分辨率页明显更准
            labs, cfs = digit_clf.digit(mp)
        else:
            labs, cfs = net.digit_with_reject(mp, min_prob=0.30)
        labs = list(labs)
        cfs = np.asarray(cfs, dtype=np.float32)
        labs, cfs = refine_by_clusters(mp, labs, cfs)
        for c, l, f in zip(music_comps, labs, cfs):
            page_labels[id(c)] = (l, float(f))

    systems: List[System] = []
    for r, row in enumerate(rows):
        if row.kind != "music":
            continue
        lyric_rows: List[Tuple[int, int]] = []
        j = r + 1
        gap_ref = max(6, int(1.2 * scale_w))
        while j < len(rows):
            nxt = rows[j]
            if nxt.kind == "music":
                break
            if nxt.kind == "lyric" and nxt.y0 - (lyric_rows[-1][1] if lyric_rows else row.y1) < 4.5 * scale_w:
                lyric_rows.append((nxt.y0, nxt.y1))
                j += 1
                continue
            break
        y1 = lyric_rows[-1][1] if lyric_rows else row.y1
        notes = build_notes(clean, row, net, scale_w, row_index=len(systems), debug=debug,
                            dash=dash_local, preset=page_labels)
        if not notes:
            warnings.append(f"行 {len(systems)} (y={row.y0}-{row.y1}) 检出 0 个音符，需人工确认")
        systems.append(System(
            row_index=len(systems), y0=row.y0, y1=y1, notes=notes, lyric_rows=lyric_rows,
            x0=min([c.x for c in row.comps], default=0),
            x1=max([c.x + c.w for c in row.comps], default=0),
        ))
    return Page(path=path, gray=gray_w, clean=clean, scale=scale_w, rows=rows, systems=systems,
                comps=comps, rules=rules, dash=dash, warnings=warnings, k=k)


def split_wide_component(clean: np.ndarray, c: Comp, scale: float):
    """把"数字+增时线/附点"粘成的宽块按垂直投影拆成若干子块。

    低分辨率谱里 `5 - -` 的增时线与数字只隔 1 px，二值化后直接粘成一块，
    宽高比 3~6，会被当成噪声丢掉——这是"整行音符没识别到"的常见原因。
    """
    if c.aspect <= 1.40:
        return [c]
    sub = clean[c.y:c.y + c.h, c.x:c.x + c.w]
    if sub.size == 0:
        return [c]
    col = (sub > 0).sum(axis=0).astype(float)
    if col.max() <= 0:
        return [c]
    # 谷底：列墨量 <= 15% 峰值，且连续谷宽 >= 1
    thr = max(1.0, 0.15 * col.max())
    cuts = []
    run = []
    for x, v in enumerate(col):
        if v <= thr:
            run.append(x)
        elif run:
            cuts.append(int(np.mean(run)))
            run = []
    if run:
        cuts.append(int(np.mean(run)))
    if not cuts:
        return [c]
    parts = []
    start = 0
    for cut in cuts + [c.w]:
        if cut - start < max(3, 0.25 * scale):
            continue
        seg = sub[:, start:cut]
        ys, xs = np.nonzero(seg)
        if len(ys) == 0:
            start = cut
            continue
        parts.append(Comp(c.x + start + int(xs.min()), c.y + int(ys.min()),
                          int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                          int(len(ys))))
        start = cut
    return parts or [c]


def refine_by_clusters(patches, labels, confs, sim: float = 0.80, min_size: int = 2):
    """页内自洽：同一页里同一个数字字形会重复出现十几次。

    把字形聚类，然后用簇内高置信成员投票定簇标签，再把标签铺回整簇。
    低分辨率页（笔画被 JPEG 撕开）靠这一招能补回大量漏检和错检。
    """
    import numpy as _np

    n = len(patches)
    if n == 0:
        return labels, confs
    masks = [_np.stack([_np.asarray(pv, dtype=_np.float32) for pv in _norm20(p)]) for p in patches]
    masks = [_np.asarray(m) for m in masks]
    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            a, b = masks[i] > 0.5, masks[j] > 0.5
            inter = _np.logical_and(a, b).sum()
            union = _np.logical_or(a, b).sum()
            if union and inter / union >= sim:
                group.append(j)
                used[j] = True
        if len(group) < min_size:
            continue
        votes = {}
        for g in group:
            lab, cf = labels[g], confs[g]
            if lab is None:
                continue
            votes[lab] = votes.get(lab, 0.0) + max(0.05, cf)
        if not votes:
            continue
        best = max(votes.items(), key=lambda kv: kv[1])[0]
        share = votes[best] / sum(votes.values())
        if share < 0.5:
            continue
        for g in group:
            if labels[g] != best:
                labels[g] = best
                confs[g] = max(float(confs[g]), 0.55)
            else:
                confs[g] = max(float(confs[g]), 0.65)
    return labels, confs


def _norm20(patch):
    import cv2 as _cv2
    import numpy as _np
    a = _np.asarray(patch)
    if a.dtype != _np.uint8:
        a = a.astype(_np.uint8)
    ys, xs = _np.nonzero(a)
    if len(ys) == 0:
        return _np.zeros((20, 20), _np.float32)
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    sc = 18.0 / max(h, w)
    res = _cv2.resize(sub, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=_cv2.INTER_AREA)
    can = _np.zeros((20, 20), _np.uint8)
    oy, ox = (20 - res.shape[0]) // 2, (20 - res.shape[1]) // 2
    can[oy:oy + res.shape[0], ox:ox + res.shape[1]] = res
    return (can > 127).astype(_np.float32)


def _overlaps_dash(dash: np.ndarray, m: Comp, pad: int = 1) -> bool:
    h, w = dash.shape
    x0, y0 = max(0, m.x - pad), max(0, m.y - pad)
    x1, y1 = min(w, m.x + m.w + pad), min(h, m.y + m.h + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    sub = dash[y0:y1, x0:x1]
    return float((sub > 0).mean()) > 0.25


def _is_dot_like(m: Comp, scale: float, clean: np.ndarray = None) -> bool:
    """八度点判据：小而**圆**的实心块。

    判据来源：OrpheusNet 的 `UpNet.py` / `DownNet.py`（公开代码）——
    - 点 vs 减时线用**形状**分：`aspect < 2 → dot，否则 dash`；
    - 点的「圆」用等周商 `周长²/(4·面积) ≈ π` 判，和分辨率无关。

    早先我用「尺寸上限 + 填充率」并把「挨着减时线的候选点」全部否掉
    （`_overlaps_dash`）——那是**自己造出来的 bug**：低音点本来就紧贴减时线，
    这一否就把真点连坐，成了漏点的系统性来源。OrpheusNet 原版没有这一步，
    因为它把点和线放同一个"下带"里用形状区分就够了。
    """
    H = max(1.0, float(scale))
    if m.h <= 0 or m.w <= 0 or m.h < 2 or m.w < 2:
        return False
    if m.h > 0.50 * H or m.w > 0.50 * H:
        return False
    asp = m.w / float(m.h)
    if asp >= 2.0:                        # 扁长 = 减时线 / 横杠 / 延音线残片
        return False
    if asp < 0.35:                        # 竖条 / 碎笔画
        return False
    if m.area / float(m.w * m.h) < 0.50:  # 空心或抗锯齿撕开
        return False
    # 圆度（OrpheusNet `UpNet` 的等周商）在这里**不用**：那是给高清扫描调的，
    # `|1 - iso/π| < 0.39`。实测 28px 字高下八度点只有 8×8，等周商≈1.66，
    # 一律被拒（为了你唱下去的点数从 38 掉到 1）。小墨块上圆度没有区分力，
    # 真正把点和减时线分开的是上面的 aspect（OrpheusNet `DownNet` 就只用这一条）。
    return True


# 纵向断笔合并的阈值（单位：字高）。低分辨率页需要更松的间隙容忍，
# 这些提到模块级是为了能扫参对比，而不是写死在函数里拍脑袋。
MERGE_GAP = 0.30
MERGE_MAXH = 1.45
MERGE_MINPART = 0.22
MERGE_SMALL = 0.42


def _split_dot_from_glyph(clean: np.ndarray, c: Comp, plain_h: float
                          ) -> Tuple[Optional[Comp], Comp]:
    """把「八度点粘连在数字上」的块拆成 (点, 干净字形)。

    这是低分辨率页的主要漏点来源：点只有 3 px，和数字的间隙 1~2 px，
    二值化后直接连成一个连通块。高度分布会呈**双峰**——
    三月雨实测 27 px（纯数字）151 个、38 px（数字+点）283 个，
    也就是说 57% 的音，点从来没进过 `marks`，根本没机会被检测。

    做法：行墨量剖面。数字主体的行很宽，点所在的行很窄。
    顶部/底部连续若干行宽度都 < 0.45×主体宽度、且厚度在 0.12~0.34 字高之间，
    就认为是点，切出去。
    """
    sub = patch_of(clean, c, pad=0)
    if sub is None or sub.size == 0 or plain_h <= 0:
        return None, c
    if c.h < 1.20 * plain_h:
        return None, c                     # 没有多余高度，不可能带点
    b = (sub > 0)
    rows = b.sum(axis=1).astype(np.float64)
    H, W = b.shape
    if H < 6:
        return None, c

    for side in ("top", "bottom"):
        seq = rows if side == "top" else rows[::-1]
        planes = b if side == "top" else b[::-1]
        # 跳过边缘空白
        i0 = 0
        while i0 < H and seq[i0] == 0:
            i0 += 1
        if i0 >= H:
            continue
        # 第一段连续有墨的行 = 边缘那个孤立小块（点）
        i1 = i0
        while i1 < H and seq[i1] > 0:
            i1 += 1
        seg_h = i1 - i0
        # 后面必须跟着**空行**，才说明它是孤立的小块而不是数字本身的一部分
        if i1 >= H or seq[i1] > 0:
            continue
        if not (2 <= seg_h <= 0.40 * H):
            continue
        band = planes[i0:i1]
        ys, xs = np.nonzero(band)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        dw, dh = x1 - x0, seg_h
        if dh < 2 or dw < 2:
            continue
        if not (0.45 <= dw / float(dh) <= 2.2):
            continue                       # 不是圆点（扁长/细高都不是）
        if dw > 0.62 * c.w:
            continue
        abs_y = (c.y + i0) if side == "top" else (c.y + H - i1)
        dot = Comp(c.x + x0, abs_y, dw, dh, int(band[:, x0:x1].sum()))
        ghost = Comp(c.x, c.y + i1, c.w, c.h - i1, c.area) if side == "top" \
            else Comp(c.x, c.y, c.w, c.h - i0, c.area)
        if ghost.h >= 0.45 * plain_h:
            return dot, ghost
    return None, c


def merge_vertical_strokes(comps: Sequence[Comp], scale: float) -> List[Comp]:
    """把纵向断开的同一字形合并。

    为什么需要：印刷体里 `4` 的上下两笔经常不连通（薄笔画 + 二值化），被当成
    两个字形各占一个框。实测「为了你唱下去」里被判成 `4` 的 15 个槽位，
    多数 `h=18` 而全页字高中位数是 `28` —— 只有 0.64 倍，等于只框住了半个字。
    让人认半截图、拿半个字训练，数据从一开始就是脏的。

    合并条件（保守，避免把八度点并进来）：
      横向重叠 >= 40%（取两者较窄者）、纵向间隙 <= 0.30 字高、
      合并后高度 <= 1.45 字高、且两部分各自都不太小（>= 0.25 字高）。
    """
    if not comps:
        return []
    items = sorted(comps, key=lambda c: (c.y, c.x))
    used = [False] * len(items)
    out: List[Comp] = []
    for i, a in enumerate(items):
        if used[i]:
            continue
        cur = Comp(a.x, a.y, a.w, a.h, a.area)
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, b in enumerate(items):
                if used[j]:
                    continue
                ov = min(cur.x + cur.w, b.x + b.w) - max(cur.x, b.x)
                if ov < 0.40 * min(cur.w, b.w):
                    continue
                gap = max(cur.y, b.y) - min(cur.y + cur.h, b.y + b.h)
                if gap > MERGE_GAP * scale:
                    continue
                y0 = min(cur.y, b.y)
                y1 = max(cur.y + cur.h, b.y + b.h)
                if (y1 - y0) > MERGE_MAXH * scale:
                    continue
                small = min(cur.h, b.h)
                if small < MERGE_MINPART * scale:
                    continue          # 太碎，不是字形的一部分
                # 方形小墨块 = 八度点，不能并进来。用形状而不是纯高度判：
                # 高度阈值正好卡在 `4` 半截（9~11 px）和八度点（8 px）之间，
                # 调哪边都会错一个。点的宽高比≈1，`4` 的半截是宽条（≈1.8）。
                sw = cur.w if cur.h == small else b.w
                if small < MERGE_SMALL * scale and 0.72 <= sw / float(small) <= 1.65:
                    continue
                x0 = min(cur.x, b.x)
                x1 = max(cur.x + cur.w, b.x + b.w)
                cur = Comp(x0, y0, x1 - x0, y1 - y0, cur.area + b.area)
                used[j] = True
                changed = True
        out.append(cur)
    return out


def _split_protrusion(clean: np.ndarray, m: Comp, scale: float) -> Optional[Comp]:
    """从「八度点粘连减时线」的宽扁块里，把凸出来的那个点单独挖出来。

    这是「音符下面一画线，低音点就没了」的真正原因：
    点和减时线在二值化后**连成一个连通块**，合并后宽高比 ≥ 2，
    于是被 `_is_dot_like` 当成减时线丢掉。删掉「挨线否决」治不了这个，
    因为问题出在**连通块已经合并**，不是判据太严。

    做法：看每一列的竖直墨量。减时线所在列墨量 ≈ 线厚（2~3 px），
    点所在列明显更厚。超过基线 1.6 倍且不占大半宽度的那些列，就是点。
    """
    sub = patch_of(clean, m, pad=0)
    if sub is None or sub.size == 0:
        return None
    col = sub.sum(axis=0).astype(np.float64)
    pos = col[col > 0]
    if pos.size == 0:
        return None
    base = float(np.median(pos))
    thr = max(base * 1.6, base + 2.0)
    idx = np.nonzero(col > thr)[0]
    if idx.size == 0:
        return None
    if idx.size > 0.55 * col.size:
        return None                     # 大半个宽度都厚，不是凸起（是真字形）
    x0, x1 = int(idx.min()), int(idx.max()) + 1
    rows = np.nonzero(sub[:, x0:x1].any(axis=1))[0]
    if rows.size == 0:
        return None
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    h, w = y1 - y0, x1 - x0
    if h < 2 or w < 2:
        return None
    if not (0.5 <= w / float(h) <= 2.0):
        return None
    if h > 0.50 * scale or w > 0.50 * scale:
        return None
    area = int(sub[y0:y1, x0:x1].sum())
    return Comp(m.x + x0, m.y + y0, w, h, area)


def build_notes(clean: np.ndarray, row: Row, net, scale: float, row_index: int = 0,
                debug: Dict = None, dash: Optional[np.ndarray] = None,
                preset: Optional[Dict] = None, keep_unknown: bool = False) -> List[Note]:
    """Digits in a music row, then accidentals (left) and octave dots (above/below).

    Assignment is mutual-nearest and conservative: a mark is only attached when its
    position is unambiguous, and an accidental must be *classified* as #/b - guessing
    from shape put a '#' on nearly every note before.
    """
    H = scale
    raw = []
    for c0 in row.comps:
        raw.extend(split_wide_component(clean, c0, H))
    # 先合并、后按字高过滤：`4` 断掉的那半截往往只有 0.35 字高，
    # 先过滤就把它扔了，合并永远看不到它（实测 h=18 一直没修好）。
    glyphs = merge_vertical_strokes([c for c in raw if c.area >= 8], H)
    # 下限回到 0.45。降到 0.30 是我上一轮的猜测性改动，实测让点/碎块混进字形，
    # 反而伤了对八度点的识别（小字行改由"几何判谱行总是执行"来捞，不靠降门槛）。
    glyphs = [c for c in glyphs if c.h >= 0.45 * H]
    # 把粘连在数字上的八度点拆出来（低分辨率页 57% 的点是粘着的，不拆就全丢）
    dots_from_glyph: List[Comp] = []
    if glyphs:
        hs_ = np.array([c.h for c in glyphs], dtype=np.float64)
        plain_h = float(np.percentile(hs_, 25)) if hs_.size else H
        g2: List[Comp] = []
        for c in glyphs:
            dot, c2 = _split_dot_from_glyph(clean, c, plain_h)
            if dot is not None:
                dots_from_glyph.append(dot)
            g2.append(c2)
        glyphs = g2
    marks = [c for c in row.comps if c.h < 0.45 * H or c.area < 8]
    # 被并进字形的碎块不能留在 marks 里，否则会被当成八度点/升降号
    if glyphs:
        _gb = [c.box for c in glyphs]
        marks = [m for m in marks
                 if not any(m.x >= b[0] - 1 and m.x + m.w <= b[2] + 1
                            and m.y >= b[1] - 1 and m.y + m.h <= b[3] + 1 for b in _gb)]
    if not glyphs:
        return []
    patches = [patch_of(clean, c) for c in glyphs]
    labels = [None] * len(glyphs)
    confs = np.zeros(len(glyphs), dtype=np.float32)
    if preset:
        for i, c in enumerate(glyphs):
            got = preset.get(id(c))
            if got is not None:
                labels[i], confs[i] = got[0], got[1]
    if net is not None:
        miss = [i for i, l in enumerate(labels) if l is None]
        if miss:
            l2, c2 = net.digit_with_reject([patches[i] for i in miss])
            for k, i in enumerate(miss):
                labels[i], confs[i] = l2[k], float(c2[k])
        # 闭集兜底：谱行里除了音符没有别的字母表（小节线/反复号已按几何剔除），
        # 分类器说"不是数字"的，只要形状像数字就拿它 0-7 里的最高分，标成低置信。
        need = [i for i, lab in enumerate(labels) if lab is None]
        if need:
            cls, prob = net.predict([patches[i] for i in need])
            labels = list(labels)
            confs = np.array(confs, dtype=np.float32)
            for k, i in enumerate(need):
                c = glyphs[i]
                if not (0.14 <= c.aspect <= 1.35):
                    continue
                scores = net.digit_scores(patches[i]) if hasattr(net, "digit_scores") else None
                name = net.classes[int(cls[k])]
                if name in "01234567":
                    labels[i] = name
                    confs[i] = min(float(prob[k]), 0.5)
                elif scores is not None:
                    j = int(np.argmax(scores))
                    if scores[j] >= 0.12:
                        labels[i] = "01234567"[j]
                        confs[i] = 0.35
    else:
        labels, confs = [None] * len(glyphs), np.zeros(len(glyphs))

    notes: List[Note] = []
    for c, lab, cf in zip(glyphs, labels, confs):
        if lab is None:
            # keep_unknown: 分类器没认出字形，但它**占了位置**。位置槽位必须保留，
            # 否则几何序列比 OCR 数字序列少几个音，按索引对齐会整体错位一位
            # （实测把 `6 1̇ 2̇ 3̇` 的八度点整体挪到 `6 1̇` 上）。
            if not keep_unknown:
                continue
            lab = "?"
        notes.append(Note(digit=lab, x=c.cx, box=c.box, conf=float(cf), row=row_index))
    if notes:
        hh = np.median([n.box[3] - n.box[1] for n in notes])
        ww = np.median([n.box[2] - n.box[0] for n in notes])
        kept = []
        for n in notes:
            w = n.box[2] - n.box[0]
            h = n.box[3] - n.box[1]
            if h < 0.55 * hh or w < 0.22 * ww:
                if debug is not None:
                    debug.setdefault("fragments", []).append(n.box)
                continue
            kept.append(n)
        notes = kept
    notes.sort(key=lambda n: n.x)
    if debug is not None:
        debug.setdefault("rejected", []).extend(
            [(c.box, "glyph") for c, lab in zip(glyphs, labels) if lab is None])
    if not notes:
        return []

    # ---- classify every mark once
    mark_patches = [patch_of(clean, m) for m in marks]
    if net is not None and mark_patches:
        l, p = net.predict(mark_patches)
        mark_cls = [(net.classes[int(a)], float(b)) for a, b in zip(l, p)]
    else:
        mark_cls = [("other", 0.0)] * len(marks)

    acc_marks = []
    dot_marks = []
    for m, (name, prob) in zip(marks, mark_cls):
        if _is_dot_like(m, H, clean):
            dot_marks.append(m)
            continue
        # 点被减时线粘住时会整块变成宽扁块 —— 按列墨量把点挖出来
        if m.w / max(1.0, float(m.h)) >= 1.6:
            got = _split_protrusion(clean, m, H)
            if got is not None and _is_dot_like(got, H, clean):
                dot_marks.append(got)
                continue
        if name in ("#", "b") and prob >= 0.60 and m.h >= 0.30 * H:
            acc_marks.append((m, name))

    # ---- accidentals: left of a note, vertically overlapping it
    for m, name in acc_marks:
        best, bestd = None, 1e9
        for n in notes:
            if n.digit == "0":
                continue
            x0, y0, x1, y1 = n.box
            dx = x0 - m.cx
            if dx <= 0 or dx > 1.1 * H:
                continue
            overlap = min(y1, m.y + m.h) - max(y0, m.y)
            if overlap < 0.5 * min(m.h, y1 - y0):
                continue
            if dx < bestd:
                best, bestd = n, dx
        if best is not None:
            best.acc = name
            best.acc_box = m.box

    # ---- octave dots: 形状已经在 _is_dot_like 里筛过（圆度 + aspect）
    # **不再按「是否挨着减时线」否决**：低音点本来就贴着线，那样会连坐把真点删掉。
    dot_cands = list(dot_marks) + list(dots_from_glyph)

    assigned = 0
    for m in dot_cands:
        best, bestd, side = None, 1e9, 0
        for n in notes:
            if n.digit == "0":
                continue
            x0, y0, x1, y1 = n.box
            # 居中容差 0.50 字宽（点常常不在正中，收紧会漏）
            if abs(m.cx - n.x) > max(0.50 * (x1 - x0), 0.30 * H):
                continue
            gap_up = y0 - (m.y + m.h)
            gap_dn = m.y - y1
            # 上侧 0.75 / 下侧 0.60 字高：低分辨率页的点离得远，收紧会漏
            if 0.02 * H <= gap_up <= 0.75 * H:
                d, sd = abs(m.cx - n.x) + 2 * gap_up, 1
            elif 0.02 * H <= gap_dn <= 0.60 * H:
                d, sd = abs(m.cx - n.x) + 2 * gap_dn, -1
            else:
                continue
            if d < bestd:
                best, bestd, side = n, d, sd
        if best is None:
            continue
        up = sum(1 for b in best.dot_boxes if b[3] <= best.box[1] + 1)
        dn = len(best.dot_boxes) - up
        if side > 0 and up >= 2:
            continue
        if side < 0 and dn >= 2:
            continue
        if (side > 0 and dn) or (side < 0 and up):
            continue                      # a note never has dots on both sides
        best.dot_boxes.append(m.box)
        best.octave += side
        assigned += 1

    # 行级保险：只在真正病态（>95%）时清。
    # 曾以为 55% 这条是在"压噪声"、并据此回退过一次 —— 那是看低分辨率预览的误判。
    # 放大 13 倍核对原图：`3 2 1` 三个音上**确实都有点**，整行高音是真实记号，
    # 55% 阈值把它们全部清空，才是"整行高音/低音全没识别"的根因。
    if notes and assigned > 0.95 * len(notes):
        for n in notes:
            n.dot_boxes.clear()
            n.octave = 0
        if debug is not None:
            debug["dot_overflow"] = debug.get("dot_overflow", 0) + 1
    return notes
