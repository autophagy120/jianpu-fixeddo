"""Two outputs: a transposed image (digits redrawn in place) and a fixed-do text score."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import music as M
from .core import Note, Page, System

DEFAULT_FONT = r"C:\Windows\Fonts\msyh.ttc"
FALLBACK_FONTS = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\times.ttf",
                  r"C:\Windows\Fonts\arialbd.ttf"]


def _font(size: int, path: str = DEFAULT_FONT) -> ImageFont.FreeTypeFont:
    for p in [path] + FALLBACK_FONTS:
        try:
            return ImageFont.truetype(p, max(6, int(size)))
        except Exception:
            continue
    raise RuntimeError("no usable font")


def _text_bbox(font, txt: str):
    return font.getbbox(txt)


def fit_font(txt: str, target_h: float, path: str = DEFAULT_FONT, max_w: Optional[float] = None) -> ImageFont.FreeTypeFont:
    """Binary-search a font size whose ink height matches `target_h`."""
    lo, hi = 6, 400
    best = _font(int(target_h), path)
    for _ in range(18):
        mid = (lo + hi) // 2
        f = _font(mid, path)
        x0, y0, x1, y1 = _text_bbox(f, txt)
        h = y1 - y0
        w = x1 - x0
        too_small = h < target_h
        if max_w is not None and w > max_w:
            too_small = False
        if too_small:
            lo = mid
            best = f
        else:
            hi = mid
        if hi - lo <= 1:
            break
    return best


NAMED_COLORS = {
    "red": (214, 30, 30),
    "blue": (30, 80, 220),
    "green": (20, 140, 60),
    "black": (0, 0, 0),
    "magenta": (190, 30, 160),
}


def parse_color(value) -> Tuple[int, int, int]:
    """'red' | '#D61E1E' | (r,g,b) -> (r,g,b)"""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    txt = str(value or "red").strip().lower()
    if txt in NAMED_COLORS:
        return NAMED_COLORS[txt]
    if txt.startswith("#") and len(txt) in (7, 4):
        h = txt[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return NAMED_COLORS["red"]


@dataclass
class ImageOptions:
    font_path: str = DEFAULT_FONT
    ink_color: Tuple[int, int, int] = (214, 30, 30)   # 红色：已识别/已重绘
    uncertain_color: Tuple[int, int, int] = (235, 140, 0)  # 橙色：待核对
    sample_ink: bool = False                          # True 则沿用原图墨色
    erase_margin: int = 1
    footer: bool = True
    footer_text: str = ""
    dot_radius_frac: float = 0.085


def _ink_color(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    x0, y0, x1, y1 = box
    patch = bgr[max(0, y0):y1, max(0, x0):x1]
    if patch.size == 0:
        return (0, 0, 0)
    g = patch.mean(axis=2)
    thr = np.percentile(g, 20)
    sel = patch[g <= thr]
    if sel.size == 0:
        return (0, 0, 0)
    b, gg, r = np.median(sel, axis=0)
    return (int(b), int(gg), int(r))


def redraw_notes(
    bgr: np.ndarray,
    page: Page,
    new_notes: Sequence[Sequence[Optional[Note]]],
    opts: ImageOptions = ImageOptions(),
    orphans: Sequence[Tuple[int, int, int, int]] = (),
) -> np.ndarray:
    """Repaint every recognised note with its transposed value.

    Two passes on purpose: erase **everything** first, then draw.  Interleaving them
    let a later note's erase-rectangle punch out a dot that an earlier note had just
    drawn (that is how a repainted 3 lost its octave dot).
    """
    k = getattr(page, "k", 1.0) or 1.0
    rgb = Image.fromarray(bgr[:, :, ::-1])          # BGR -> RGB
    draw = ImageDraw.Draw(rgb)
    W, H = rgb.size
    m = opts.erase_margin

    def px(v):
        return int(round(v / k))

    # ---- pass 1: erase
    draw_list = []
    for si, sysn in enumerate(page.systems):
        if si >= len(new_notes):
            break
        for ni, note in enumerate(sysn.notes):
            repl = new_notes[si][ni] if ni < len(new_notes[si]) else None
            if repl is None:
                continue
            if not getattr(repl, "confirmed", True):
                repl = repl
            x0, y0, x1, y1 = (px(v) for v in note.box)
            w, h = max(1, x1 - x0), max(1, y1 - y0)
            draw.rectangle([x0 - m, y0 - m, x1 + m, y1 + m], fill=(255, 255, 255))
            if note.acc_box is not None:
                ab = tuple(px(v) for v in note.acc_box)
                draw.rectangle([ab[0] - m, ab[1] - m, ab[2] + m, ab[3] + m], fill=(255, 255, 255))
            for db in note.dot_boxes:
                d = tuple(px(v) for v in db)
                draw.rectangle([d[0] - m, d[1] - m, d[2] + m, d[3] + m], fill=(255, 255, 255))
            # a dot may hide slightly higher/lower than the detected box: clear the band
            band = int(round(0.42 * h))
            draw.rectangle([x0 + int(0.25 * w), y0 - band, x1 - int(0.25 * w), y0 - 1], fill=(255, 255, 255))
            draw.rectangle([x0 + int(0.25 * w), y1 + 1, x1 - int(0.25 * w), y1 + band], fill=(255, 255, 255))
            draw_list.append((note, repl, (x0, y0, x1, y1)))

    # ---- 统一字号：取所有已识别数字高（原图坐标）的中位数
    heights = []
    for note, repl, (x0, y0, x1, y1) in draw_list:
        heights.append(y1 - y0)
    if heights:
        heights.sort()
        unit_h = heights[len(heights) // 2]
    else:
        unit_h = 0
    widths = sorted(x1 - x0 for _n, _r, (x0, _y0, x1, _y1) in draw_list) or [0]
    unit_w = widths[len(widths) // 2]

    # ---- pass 2: draw
    for note, repl, (x0, y0, x1, y1) in draw_list:
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        if opts.sample_ink:
            ink = _ink_color(bgr, tuple(px(v) for v in note.box))
        elif getattr(repl, "confirmed", True):
            ink = opts.ink_color
        else:
            ink = opts.uncertain_color
        txt = repl.digit
        # 统一字号：以全页中位数字高为准，只有极端离群（被切坏/粘连）才按本字高度兜底
        hh = unit_h or h          # 全页统一字号：碎块/粘连块不再把字放大或缩小
        f = fit_font(txt, 0.96 * hh, opts.font_path, max_w=max(6.0, 1.45 * unit_w))
        bb = _text_bbox(f, txt)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        draw.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), txt, font=f, fill=ink)

        if repl.acc:
            fa = fit_font(repl.acc, 0.92 * hh, opts.font_path)
            ba = _text_bbox(fa, repl.acc)
            aw, ah = ba[2] - ba[0], ba[3] - ba[1]
            # 贴在数字左侧、与数字同高居中；字母压在数字上时稍微左移，宁可侵入空白
            ax = x0 - 0.26 * unit_w - aw - ba[0]
            ay = cy - ah / 2 - ba[1]
            draw.text((ax, ay), repl.acc, font=fa, fill=ink)

        r = max(1.2, opts.dot_radius_frac * hh)
        if repl.octave > 0:
            for i in range(min(repl.octave, 2)):
                yc = y0 - 0.26 * hh - i * 0.5 * hh
                draw.ellipse([cx - r, yc - r, cx + r, yc + r], fill=ink)
        elif repl.octave < 0:
            for i in range(min(-repl.octave, 2)):
                yc = y1 + 0.26 * hh + i * 0.5 * hh
                draw.ellipse([cx - r, yc - r, cx + r, yc + r], fill=ink)

    # 未识别字形：橙色虚框 + 问号（这些位置有音符，但没认出来）
    for (ox0, oy0, ox1, oy1) in orphans:
        draw.rectangle([ox0, oy0, ox1, oy1], outline=opts.uncertain_color, width=1)
        f = _font(max(8, int(0.9 * (oy1 - oy0))), opts.font_path)
        draw.text((ox0, oy0 - 0.9 * (oy1 - oy0)), "?", font=f, fill=opts.uncertain_color)

    if opts.footer and opts.footer_text:
        f = _font(max(10, int(0.016 * H)), opts.font_path)
        bb = _text_bbox(f, opts.footer_text)
        tw = bb[2] - bb[0]
        draw.text((max(4, (W - tw) / 2 - bb[0]), 6 - bb[1]), opts.footer_text, font=f, fill=(120, 120, 120))

    return np.array(rgb)[:, :, ::-1].copy()


# ------------------------------------------------------------------ text out
@dataclass
class TextOptions:
    style: str = "cn"              # cn: （1）【1】  ascii: (1)[1]
    harmonica: bool = True
    annotate: bool = False
    show_key_header: bool = True
    bars: bool = False


def render_text(
    page: Page,
    key_pc: int,
    ref_pc: int,
    ref_reason: str,
    text_opts: TextOptions = TextOptions(),
    harm: Optional[M.HarmonicaOptions] = None,
    title: str = "",
    bars_per_phrase: int = 4,
) -> str:
    from . import phrases as PH

    harm = harm or M.HarmonicaOptions()
    lines: List[str] = []
    if text_opts.show_key_header:
        lines.append(f"# {title or os.path.basename(page.path)}")
        lines.append(f"# 原调 1={M.key_name(key_pc, prefer_flat=True)}   固定调 1={M.key_name(ref_pc)}   {ref_reason}")
        if text_opts.harmonica:
            lines.append("# 半音阶口琴优化已开启（等音替换以减少换气/推键切换）")
        stats = [f"行 {len(page.systems)}"]
        lines.append("# " + "  ".join(stats))
        lines.append("")
    for si, sysn in enumerate(page.systems):
        if not sysn.notes:
            lines.append(f"# [第{si + 1}行] 无音符，需人工确认")
            continue
        phs, how = PH.split_notes(page, sysn, page.scale, bars_per_phrase)
        pitches = []
        for n in sysn.notes:
            if n.digit == "0":
                pitches.append(None)
            else:
                pitches.append(M.note_to_abs(int(n.digit), n.acc, n.octave, key_pc))
        for (a, b) in phs:
            seg = [p for p in pitches[a:b]]
            idx = [i for i, p in enumerate(seg) if p is not None]
            if not idx:
                continue
            toks = [""] * len(seg)
            rendered = M.render_fixed_do(
                [seg[i] for i in idx], ref_pc, harm, text_opts.style,
                harmonica=text_opts.harmonica, annotate=text_opts.annotate,
            )
            for t, i in zip(rendered, idx):
                toks[i] = t
            for i, p in enumerate(seg):
                if p is None:
                    toks[i] = "0"
            lines.append(" ".join(toks))
    return "\n".join(lines) + "\n"
