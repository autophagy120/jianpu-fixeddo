"""在原谱图片上按坐标把音符替换成固定调红字。

为什么需要这条路（而不是纯文本谱）：
简谱里有大量**文字谱线性写法表达不了**的结构——
- `( ... ) :||` 反复段：音符只印一次，唱两遍；
- 一条谱行配**两行歌词**（第一段/第二段），文字谱只能把音符行复制一遍；
- 反复时并行的两段歌词、`D.S.`、`跳房子`。

只要保持"音符在原来的位置上"，这些结构自动保留。所以我们只做一件事：
把每个音符**就地**换成固定调的写法（红字），版面、歌谱、反复记号、歌词都不动。

实现要点：
- 坐标全部用 `core.Page` 的工作空间（`page.gray`/`page.clean` 同一坐标系），
  槽位 `Note.box` 就是这么来的，不跨坐标系换算；
- 擦除用**矩形白化**，只覆盖字形自身的框 + 八度点框 + 升降号框。
  减时线的横杠在字形框外（下方独立连通块），不会被误擦；
- 新写法长度不定（`【3】` 比 `3` 宽），所以字号会按前后邻居的可用空档收缩，
  实在放不下时画橙色 `?` 并计数——不静默画错。
"""
from __future__ import annotations

import functools
import hashlib
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core
from . import music as M
from . import xiaomi_ocr as XO

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]


@functools.lru_cache(maxsize=256)
def _font(size: int) -> ImageFont.FreeTypeFont:
    """按字号缓存字体对象。

    原来每个音都 `truetype()` 一次：一页 212 个音 = 212 次打开字体文件，
    实测占一次重画的 0.15s，纯浪费。
    """
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, max(8, int(size)))
            except Exception:
                continue
    raise RuntimeError("找不到可用字体，请在 _FONT_CANDIDATES 里补一个")


def _text_size(font: ImageFont.FreeTypeFont, s: str) -> Tuple[int, int]:
    box = font.getbbox(s)
    return box[2] - box[0], box[3] - box[1]


def _erase_boxes(gray: np.ndarray, boxes: Sequence[core.Box], pad: int = 1) -> None:
    """矩形白化（就地修改）。"""
    h, w = gray.shape
    for (x0, y0, x1, y1) in boxes:
        a = max(0, int(x0) - pad)
        b = max(0, int(y0) - pad)
        c = min(w, int(x1) + pad)
        d = min(h, int(y1) + pad)
        if c > a and d > b:
            gray[b:d, a:c] = 255


def _avail_span(slots: Sequence[core.Note], i: int) -> Tuple[float, float]:
    """第 i 个槽位左右两侧可用的水平空档（不侵犯邻居的墨迹）。"""
    s = slots[i]
    x0, _, x1, _ = s.box
    left = slots[i - 1].box[2] if i > 0 else x0 - 2 * (x1 - x0)
    right = slots[i + 1].box[0] if i + 1 < len(slots) else x1 + 2 * (x1 - x0)
    return float(left), float(right)


def _erase_stray_dots(page: core.Page, pairs, gray: np.ndarray) -> int:
    """擦掉谱行数字带之外的、点状的小墨块（没被指派的八度点）。

    必须**按谱行**限定范围，否则会连歌词和其他行的墨块一起擦掉
    （实测全页遍历会擦 600~1400 块，把版面啃烂）。
    规则：只处理落在「本行数字带上方/下方 1.1 个字高以内、且 x 在本行范围内」
    的小块；附点 `·` 在数字带内部，属于节奏记号，保留。
    """
    if not pairs:
        return 0
    by_row: dict = {}
    for s, _o in pairs:
        by_row.setdefault(int(s.row), []).append(s.box)
    n = 0
    for _row, boxes in by_row.items():
        x0 = min(b[0] for b in boxes)
        x1 = max(b[2] for b in boxes)
        yt = min(b[1] for b in boxes)
        yb = max(b[3] for b in boxes)
        med_h = float(np.median([b[3] - b[1] for b in boxes]))
        for c in page.comps:
            if c.area < 6 or c.h < 2 or c.w < 2:
                continue
            if c.h > 0.55 * med_h or c.w > 0.55 * med_h:
                continue
            if c.w > 2.0 * c.h:
                continue          # 扁长 = 减时线/横杠残片，绝不能擦
            if c.x + c.w < x0 or c.x > x1:
                continue
            # 完全在数字带上方或下方（留 1px 余量防啃字形边缘）
            above = (c.y + c.h < yt - 1)
            below = (c.y > yb + 1)
            if not (above or below):
                continue
            # 且必须在带子附近，别跑到歌词行去
            cy = c.y + c.h / 2.0
            if cy < yt - 1.1 * med_h or cy > yb + 1.1 * med_h:
                continue
            gray[max(0, c.y - 1):c.y + c.h + 1,
                 max(0, c.x - 1):c.x + c.w + 1] = 255
            n.append((y0, y1, x0, x1))
    return n


def _fill_uncovered(page: core.Page, comps: Sequence[core.Comp], gray: np.ndarray,
                    img: "Image.Image", draw: "ImageDraw.ImageDraw",
                    med_h: float, ink_color, font_h: float) -> int:
    """把"几何看得见、但没配上数字"的墨块，用 RapidOCR 的带框结果补成红字。

    针对的就是 `(3) 3· 4 4 1 |` 这类**小字号和声/装饰行**：几何侧切出来了，
    但 OCR 的文本层要么把它并进相邻行、要么整行省略，于是这些音只能标橙框。
    这里按坐标把它们和 RapidOCR 的字符配对，补画成红字（顺带补八度点）。
    """
    if not comps:
        return 0
    try:
        from . import ocr_boxes as OB
        import re as _re
        stem = _re.sub(r"[^\w\u4e00-\u9fff]+", "_", os.path.basename(page.path))[:70]
        chars, _s = OB.read_page_chars(page.gray, os.path.join("out", "_ocrboxes", stem + ".json"))
    except Exception:
        return 0
    if not chars:
        return 0
    used = [False] * len(chars)
    cand = []
    for ci, c in enumerate(comps):
        cx = c.x + c.w / 2.0
        cy = c.y + c.h / 2.0
        for k, ch in enumerate(chars):
            dx = abs(ch.cx - cx)
            dy = abs(ch.cy - cy)
            if dx <= 0.65 * med_h and dy <= 0.65 * med_h:
                cand.append((dx + dy, ci, k))
    cand.sort()
    done = set()
    n = 0
    for _d, ci, k in cand:
        if ci in done or used[k]:
            continue
        done.add(ci)
        used[k] = True
        c = comps[ci]
        ch = chars[k].ch
        gh = max(float(c.h), 0.92 * font_h)
        font = _font(int(round(gh)))
        bbox = font.getbbox(ch)
        ox = (c.x + c.w / 2.0) - (bbox[0] + bbox[2]) / 2.0
        oy = (c.y + c.h) - bbox[3]
        draw.text((ox, oy), ch, font=font, fill=ink_color)
        # 八度点：数一下紧邻上下的小墨块
        r = max(2.0, 0.125 * gh)
        cx = c.x + c.w / 2.0
        for side in (+1, -1):
            nb = 0
            for d in page.comps:
                if d is c or d.h < 2 or d.w < 2 or d.h > 0.5 * med_h or d.w > 0.5 * med_h:
                    continue
                if d.w > 2.0 * d.h:
                    continue
                if abs((d.x + d.w / 2.0) - cx) > 0.45 * max(1.0, c.w):
                    continue
                gap = (c.y - (d.y + d.h)) if side > 0 else (d.y - (c.y + c.h))
                if 0.02 * med_h <= gap <= 0.55 * med_h:
                    nb += 1
                    if nb <= 2:
                        dy = (c.y - 0.10 * gh - r - (nb - 1) * 2.2 * r) if side > 0 \
                            else (c.y + c.h + 0.10 * gh + r + (nb - 1) * 2.2 * r)
                        draw.ellipse([cx - r, dy - r, cx + r, dy + r], fill=ink_color)
            if nb >= 2:
                break
        n.append((y0, y1, x0, x1))
    return n


def erase_unpaired_rows(page, pairs, gray, wipe: bool = False):
    """没配上任何音符的谱行：默认**不擦**，只返回行带用于标记。

    以前这里会把整行擦白，结果一旦对齐失败（比如 OCR 少了那几行），
    整行原始谱面就被抹掉了 —— 用户看到"一整行都被擦掉、只剩一个橙框"，
    而且删音符也救不回来（原墨没了）。现在改成只画橙色标记，保留原谱。
    """
    if not pairs:
        return []
    boxes = [(s.box[1], s.box[3]) for s, _o in pairs]
    n = []
    for r in page.rows:
        if r.kind != "music":
            continue
        hit = any(min(b[1], r.y1) - max(b[0], r.y0) > 0.3 * (r.y1 - r.y0) for b in boxes)
        if hit:
            continue
        x0 = max(0, min((c.x for c in r.comps), default=0) - 3)
        x1 = min(gray.shape[1], max((c.x + c.w for c in r.comps), default=gray.shape[1]) + 3)
        y0 = max(0, int(r.y0) - 2)
        y1 = min(gray.shape[0], int(r.y1) + 2)
        if wipe:
            gray[y0:y1, x0:x1] = 255
        n.append((y0, y1, x0, x1))
    return n

_BG_CACHE: dict = {}
_BG_CACHE_MAX = 2


def _bg_sig(page: core.Page, pairs, erase_mode: str, med_h: float, wipe: bool = False):
    """擦除结果只取决于**几何**（槽位框/点框/升降号框），与数字、字号、调号无关。"""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray([len(pairs), med_h], dtype=np.float64).tobytes())
    h.update(np.asarray(page.gray.shape, dtype=np.int64).tobytes())
    h.update(str(erase_mode).encode("utf8"))
    h.update(b"wipe1" if wipe else b"wipe0")
    for s, _o in pairs:
        h.update(np.asarray(s.box, dtype=np.float64).tobytes())
        h.update(np.asarray([float(s.row)], dtype=np.float64).tobytes())
        for b in s.dot_boxes:
            h.update(np.asarray(b, dtype=np.float64).tobytes())
        if s.acc_box is not None:
            h.update(np.asarray(s.acc_box, dtype=np.float64).tobytes())
    return (str(getattr(page, "path", "")), h.hexdigest())


def _build_background(page: core.Page, pairs, erase_mode: str, med_h: float,
                      comps, wipe: bool = False) -> Tuple[np.ndarray, list]:
    """把原谱擦成"只剩减时线/小节线/歌词"的底图，返回 (gray, 未配对行带)。

    这是整条渲染里最贵的一步（212 个音 × 1866 个连通块）。数字/八度/字号/调号
    的修改都不改变几何，所以按几何签名缓存；只有插入/删除/回填音符才会重算。
    """
    sig = _bg_sig(page, pairs, erase_mode, med_h, wipe)
    hit = _BG_CACHE.get(sig)
    if hit is not None:
        return hit[0].copy(), list(hit[1])

    gray = page.gray.copy()
    if erase_mode == "band":
        # 按**格子**擦：每个音占一个 x 单元（到相邻音中点） + 整条谱行带的高度。
        # 这样即使槽位框只框到字形一角、或者残留笔画在框外，也一定会被擦干净。
        # 代价：该单元里的减时线（横杠）会被一并擦掉。
        band_rows: dict = {}
        for s, o in pairs:
            band_rows.setdefault(int(s.row), []).append(s)
        for _r, ss in band_rows.items():
            ss.sort(key=lambda s: s.box[0])
            y0 = min(s.box[1] for s in ss) - 0.10 * med_h
            y1 = max(s.box[3] for s in ss) + 0.10 * med_h
            for i, s in enumerate(ss):
                half = 0.55 * med_h
                xa = s.box[0] - half if i == 0 else max(s.box[0] - half,
                                                        (ss[i - 1].box[2] + s.box[0]) / 2.0)
                xb = s.box[2] + half if i == len(ss) - 1 else min(s.box[2] + half,
                                                                  (s.box[2] + ss[i + 1].box[0]) / 2.0)
                _erase_boxes(gray, [(xa, y0, xb, y1)], pad=0)

    # 连通块几何一次性转成 numpy：原来每个音都遍历全部 comps（212×1866 次 Python
    # 循环、每次 3 个 min/max），实测占 0.5s，是最大的一块开销。
    if comps:
        ca = np.array([(c.x, c.y, c.x + c.w, c.y + c.h, c.w, c.h) for c in comps],
                      dtype=np.float64)
        keep = (ca[:, 5] >= 0.18 * med_h) & (ca[:, 4] <= 2.5 * ca[:, 5])
        cx0, cy0, cx1, cy1 = ca[:, 0], ca[:, 1], ca[:, 2], ca[:, 3]
        cw, ch = ca[:, 4], ca[:, 5]
    else:
        keep = np.zeros((0,), dtype=bool)
        cx0 = cy0 = cx1 = cy1 = cw = ch = np.zeros((0,), dtype=np.float64)

    for slot, o in pairs:
        bx = slot.box
        # 外扩 0.45 字高再找重叠块：数字被分割成两半时，另一半的中心往往落在
        # 原框之外（只与框轻微重叠甚至不重叠），只按原框找就会漏擦，留下黑色残笔。
        ex = 0.45 * med_h
        ebx = (bx[0] - ex, bx[1] - ex, bx[2] + ex, bx[3] + ex)
        if len(keep):
            inside = ((cx0 >= ebx[0]) & (cx1 <= ebx[2])
                      & (cy0 >= ebx[1]) & (cy1 <= ebx[3]))
            ox = np.minimum(cx1, bx[2]) - np.maximum(cx0, bx[0])
            oy = np.minimum(cy1, bx[3]) - np.maximum(cy0, bx[1])
            overlap = (ox > 0) & (oy > 0) & (ox * oy >= 0.20 * (cw * ch))
            for i in np.nonzero(keep & (inside | overlap))[0]:
                _erase_boxes(gray, [comps[i].box], pad=1)
        extra = list(slot.dot_boxes)
        if slot.acc_box is not None:
            extra.append(slot.acc_box)
        if extra:
            _erase_boxes(gray, extra, pad=1)

    # 未配对的行整行擦白（必须在绘制**之前**做，否则重建 img 会把红字抹掉）
    try:
        bands = list(erase_unpaired_rows(page, pairs, gray, wipe=wipe))
    except Exception:
        bands = []

    if len(_BG_CACHE) >= _BG_CACHE_MAX:
        _BG_CACHE.clear()
    _BG_CACHE[sig] = (gray.copy(), list(bands))
    return gray, bands


def render_fixed_image(page: core.Page, pairs: Sequence[Tuple[core.Note, XO.OcrNote]],
                       key_pc: int, ref_pc: int,
                       ink_color: Tuple[int, int, int] = (200, 0, 0),
                       miss_color: Tuple[int, int, int] = (235, 140, 0),
                       harmonica: bool = False,
                       mark_missed: bool = True,
                       erase_mode: str = "comp",
                       keys: Optional[Sequence[int]] = None,
                       aux_ratio: float = 0.72,
                       wipe_unpaired: bool = False) -> Tuple[np.ndarray, dict]:
    """返回 (彩色 BGR 图, 统计)。

    **图片版用简谱原生写法**（数字 + 升降号 + 上下八度点），不用 `【3】`：
    `【3】` 在 30 px 字号下宽 78 px，而原数字只有 26 px，就地塞不下
    （实测每个音都要缩到最小，还留 9 处重叠）。简谱本来就用八度点表示八度，
    原生写法宽度和原字形一致，版面不会被撑坏——参考实现 `replaceimgTone`
    也是这么做的。`【】/（）` 只用在纯文本谱里。

    `keys`：**逐音调号**（长度同 `pairs`，给每个音自己的 `1=?` 音级）。
    用于谱面中途转调（`转1=G`）：转调点之后的音按新调换算，之前的按原调。
    `None` = 全谱一个调，行为与旧版完全一致。
    """
    harm = M.HarmonicaOptions()

    # 按行分组做可用空档计算（邻居只在同一行内才成立）
    by_row: dict = {}
    for k, (slot, _o) in enumerate(pairs):
        by_row.setdefault(int(slot.row), []).append(k)

    comps = page.comps
    med_h = float(np.median([max(1.0, s.box[3] - s.box[1]) for s, _o in pairs]
                            or [1.0])) if pairs else 1.0
    erased = 0
    # 底图（擦掉原音符）走缓存：几何没变就直接复用，省掉最贵的擦除扫描
    gray, _unpaired_bands = _build_background(page, pairs, erase_mode, med_h, comps,
                                              wipe=wipe_unpaired)
    _unpaired_n = len(_unpaired_bands)

    # ---- 行基准：同一行所有音共用**一个字号 + 一个基线**
    # 以前每个音按自己的槽位框底边对齐，框被切得深浅不一，图上就是"一个上一个下"。
    # 行内中位字高 / 中位底边对个别坏框免疫；只有标成"辅助字号"的音才破例变小。
    row_gh: dict = {}
    row_base: dict = {}
    for _row, _idxs in by_row.items():
        hs, bs = [], []
        for _k in _idxs:
            _s, _o = pairs[_k]
            if str(getattr(_s, "size_class", "main")) == "aux":
                continue                       # 例外不参与行基准
            hs.append(max(1.0, _s.box[3] - _s.box[1]))
            bs.append(float(_s.box[3]))
        if not hs:                             # 整行都是例外 → 退回整行中位
            for _k in _idxs:
                _s, _o = pairs[_k]
                hs.append(max(1.0, _s.box[3] - _s.box[1]))
                bs.append(float(_s.box[3]))
        _g = float(np.median(hs))
        if med_h > 0:                          # 防止个别行被坏框带跑
            _g = min(1.25 * med_h, max(0.80 * med_h, _g))
        row_gh[_row] = _g
        row_base[_row] = float(np.median(bs))

    rgb = np.stack([gray] * 3, axis=-1)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for (fy0, fy1, fx0, fx1) in _unpaired_bands:
        draw.rectangle([fx0, fy0, fx1 - 1, fy0 + 3], fill=miss_color)
        draw.rectangle([fx0, fy1 - 3, fx1 - 1, fy1], fill=miss_color)


    # ---- 第二遍：逐音画简谱原生写法
    stats = dict(drawn=0, rest=0, shrunk=0, overflow=0, missing=0, total=len(pairs),
                 stray_dots=erased)
    stats["unpaired_rows"] = _unpaired_n
    for row, idxs in by_row.items():
        idxs.sort(key=lambda k: pairs[k][0].x)
        for pos, k in enumerate(idxs):
            slot, o = pairs[k]
            x0, y0, x1, y1 = slot.box
            # 行内统一字号 + 统一基线（例外：标了辅助字号的音按比例缩小）
            gh = row_gh.get(row) or max(1.0, float(y1 - y0))
            base = row_base.get(row, float(y1))
            if str(getattr(slot, "size_class", "main")) == "aux":
                gh = gh * float(aux_ratio)
                stats["aux"] = stats.get("aux", 0) + 1
            # 休止符 `0` 不参与音高转换，但照样变红——它也是谱面记号，
            # 黑着会让人以为这一处没处理。
            if str(o.digit) == "0":
                font = _font(int(round(gh * 1.02)))
                bbox = font.getbbox("0")
                draw.text(((x0 + x1) / 2.0 - (bbox[0] + bbox[2]) / 2.0,
                           base - bbox[3]), "0", font=font, fill=ink_color)
                stats["rest"] += 1
                stats["drawn"] += 1
                continue
            try:
                k_pc = int(keys[k]) if keys is not None else key_pc
                p_abs = M.note_to_abs(int(o.digit), o.acc, int(o.octave), k_pc)
                tok = M.render_fixed_do(
                    [p_abs],
                    ref_pc, harm, "cn", harmonica=harmonica, octave_bias=0)[0]
            except Exception:
                k_pc, tok = key_pc, ""
            core_tok, _abs_oct = _split_token(tok)
            if not core_tok:
                stats["missing"] += 1
                continue
            # **八度用源谱自己的点**（`o.octave`），不能用绝对音高推。
            # `note_to_abs` 里"八度 0"的 1..7 度本身就跨到高八度去
            # （G 调的 `6` = E5，绝对音高就落在 【】 里），照它画会给原谱
            # 没有点的音也加上点——实测整行点全挪位。图片版是"原位替换"，
            # 点必须和原来的位置一一对应。
            octv = int(o.octave)
            # 只有**转调段**才改用绝对音高反推八度：原调段保持"沿用原谱八度点"
            # （原位替换，点必须一一对应）。转调后同一个音在新调里的度数变了，
            # 沿用原点数就会差八度，必须按绝对音高重算。
            if keys is not None and k_pc != key_pc:
                try:
                    _d, _a, octv = M.abs_to_movable(p_abs, ref_pc)
                except Exception:
                    octv = int(o.octave)

            prev_k = idxs[pos - 1] if pos > 0 else None
            next_k = idxs[pos + 1] if pos + 1 < len(idxs) else None
            left = pairs[prev_k][0].box[2] if prev_k is not None else x0 - 2.0 * gh
            right = pairs[next_k][0].box[0] if next_k is not None else x1 + 2.0 * gh
            avail = max(gh * 1.05, float(right - left) - 1.0)

            size = int(round(gh * 1.02))
            font = _font(size)
            tw, _th = _text_size(font, core_tok)
            # 缩小下限抬到 0.85：之前允许缩到 0.70，会出现"某个音突然变小"的
            # 字号误识别观感。宁可略微挤占邻格，也不让字号跳变。
            while tw > avail and size > int(gh * 0.85):
                size -= 1
                font = _font(size)
                tw, _th = _text_size(font, core_tok)
                stats["shrunk"] += 1
            if tw > avail * 1.25:
                stats["overflow"] += 1
                font = _font(int(gh * 0.9))
                draw.text((x0, base - gh), "?", font=font, fill=miss_color)
                continue

            cx = (x0 + x1) / 2.0
            bbox = font.getbbox(core_tok)
            ox = cx - (bbox[0] + bbox[2]) / 2.0
            oy = base - bbox[3]                  # 行基线对齐，不再跟各自槽位框底边走
            draw.text((ox, oy), core_tok, font=font, fill=ink_color)

            # 八度点：**跟着画出来的数字走**（字号/基线统一后，不能再按原槽位框算，
            # 否则点会飘到离数字很远的地方）。
            # 实测原图：数字 450..478（28px），点 481..488（7px）——
            # 即 gap = 0.11 字高、点径 = 0.25 字高。
            r = max(2.0, 0.125 * gh)
            if octv != 0:
                n = min(2, abs(int(octv)))
                step = 2.2 * r
                gtop = oy + bbox[1]
                gbot = oy + bbox[3]
                for i in range(n):
                    if octv > 0:
                        dy = gtop - 0.10 * gh - r - i * step
                    else:
                        dy = gbot + 0.10 * gh + r + i * step
                    if octv < 0:
                        draw.ellipse([cx - r - 1.2, dy - r - 1.2,
                                      cx + r + 1.2, dy + r + 1.2], fill=(255, 255, 255))
                    draw.ellipse([cx - r, dy - r, cx + r, dy + r], fill=ink_color)
            stats["drawn"] += 1

    if mark_missed:
        missed = find_uncovered(page, pairs)
        if missed:
            mark_uncovered(img, missed, miss_color, page.scale)
        stats["uncovered"] = len(missed)
        # 先把漏识别的墨块用 RapidOCR 补成红字，剩下的才画橙框。
        # 注意：`_fill_uncovered` 目前**必然抛 AttributeError**（该函数里 `n = 0`
        # 之后又 `n.append(...)`），异常被这里吞掉 → `filled` 恒为 0，
        # 也就是"自动补字"这条路径实际上没生效，橙框仍靠人工回填（功能 A）。
        # 是否启用属于产品决策（补字=多画红字，但那是 OCR 判断、不是人工确认），
        # 先保持现状，不悄悄改输出。
        try:
            filled = _fill_uncovered(page, missed, gray, img, draw, med_h, ink_color, med_h)
        except Exception:
            filled = 0
        stats["filled"] = filled

    bgr = cv2_cvt(img)
    return bgr, stats


def find_uncovered(page: core.Page, pairs: Sequence[Tuple[core.Note, XO.OcrNote]],
                   y_tol: float = 1.0) -> List[core.Comp]:
    """谱行上"长得像音符、但没被任何槽位覆盖"的墨块 = 漏识别的音。

    用户反馈：`为了你唱下去` 行距不标准，歌词行中间偶尔有一两个音没被识别
    （原版软件也有同样问题）。与其让它黑着混在歌词里，不如圈出来让人一眼看到。

    判据：数字形（字高和谱行音符接近、宽高比像数字）且 y 落在某个已配对槽位
    的同一行内（±`y_tol` 个字高），但不在任何槽位框里。
    """
    H = float(page.scale)
    if not pairs:
        return []
    boxes = [s.box for s, _o in pairs]
    # 升降号和八度点虽然不在字形框里，但已经被处理过了，不能算漏识别
    extra: List[core.Box] = []
    for s, _o in pairs:
        extra.extend(s.dot_boxes)
        if s.acc_box is not None:
            extra.append(s.acc_box)
    hs = np.array([b[3] - b[1] for b in boxes], dtype=np.float64)
    med_h = float(np.median(hs)) if len(hs) else H
    rows_y = np.array([(b[1] + b[3]) / 2.0 for b in boxes], dtype=np.float64)
    all_boxes = list(boxes) + extra
    out: List[core.Comp] = []
    for c in page.comps:
        # 下限 0.40（原 0.62）：小字号和声行只有主音符一半高，0.62 会把它们
        # 排除在"漏识别"之外 —— 既不补画、也不标橙框，等于凭空消失。
        if c.area < 24 or c.h < 0.40 * med_h or c.h > 1.6 * med_h:
            continue
        if core.is_rule_like(c, H):
            continue                      # 小节线 / 反复号 / 减时线，不是音符
        asp = c.w / max(1.0, float(c.h))
        if not (0.35 <= asp <= 1.35):
            continue
        cy = c.y + c.h / 2.0
        if float(np.min(np.abs(rows_y - cy))) > y_tol * max(H, med_h):
            continue
        cx0, cy0, cx1, cy1 = c.box
        covered = False
        for (x0, y0, x1, y1) in all_boxes:
            if cx0 >= x0 - 2 and cx1 <= x1 + 2 and cy0 >= y0 - 2 and cy1 <= y1 + 2:
                covered = True
                break
        if not covered:
            out.append(c)
    return out


def mark_uncovered(img: Image.Image, comps: Sequence[core.Comp],
                   color: Tuple[int, int, int] = (0, 100, 235),
                   scale: float = 1.0) -> int:
    """给漏识别的墨块画橙色虚线框（就地修改）。"""
    draw = ImageDraw.Draw(img)
    pad = max(2, int(round(0.10 * scale)))
    step = max(3, int(round(0.18 * scale)))
    for c in comps:
        x0, y0 = c.x - pad, c.y - pad
        x1, y1 = c.x + c.w + pad, c.y + c.h + pad
        # 上下左右四条虚线
        for x in range(x0, x1, step * 2):
            draw.line([(x, y0), (min(x + step, x1), y0)], fill=color, width=1)
            draw.line([(x, y1), (min(x + step, x1), y1)], fill=color, width=1)
        for y in range(y0, y1, step * 2):
            draw.line([(x0, y), (x0, min(y + step, y1))], fill=color, width=1)
            draw.line([(x1, y), (x1, min(y + step, y1))], fill=color, width=1)
    return len(comps)


_CORE_DIGITS = set("01234567")


def _split_token(tok: str) -> Tuple[str, int]:
    """把 `【3】` / `(1)` / `#4` / `7` 拆成 (简谱写法, 八度)。

    简谱写法 = 升降号 + 数字；八度 = +1 高、(0) 原、-1 低。
    """
    if not tok:
        return "", 0
    octv = 0
    s = tok
    while True:
        if s.startswith("【【") and s.endswith("】】"):
            octv += 2
            s = s[2:-2]
        elif s.startswith("【") and s.endswith("】"):
            octv += 1
            s = s[1:-1]
        elif s.startswith("（（") and s.endswith("））"):
            octv -= 2
            s = s[2:-2]
        elif s.startswith("（") and s.endswith("）"):
            octv -= 1
            s = s[1:-1]
        elif s.startswith("((") and s.endswith("))"):
            octv -= 2
            s = s[2:-2]
        elif s.startswith("(") and s.endswith(")"):
            octv -= 1
            s = s[1:-1]
        else:
            break
    s = s.strip()
    if not s:
        return "", octv
    if s[-1] not in _CORE_DIGITS:
        return "", octv
    return s, octv


def cv2_cvt(img: Image.Image) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
