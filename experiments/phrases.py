"""Split notes into phrases: lyric punctuation first, barlines as fallback."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .core import Page, System


def _lyric_comps(page: Page, system: System) -> List["object"]:
    if not system.lyric_rows:
        return []
    y0 = system.lyric_rows[0][0]
    y1 = system.lyric_rows[-1][1]
    return page.comps_in(y0, y1)


def punctuation_marks(page: Page, system: System, scale: float) -> List[Tuple[float, int]]:
    """x positions of punctuation in the lyric rows of one system.

    Chinese lyric characters are square and tall; ，。、；：！？（） are much smaller and
    sit at the baseline, so geometry alone finds them - no Chinese OCR needed.
    """
    comps = [c for c in _lyric_comps(page, system) if c.h >= 4 and c.area >= 6]
    if not comps:
        return []
    hs = np.array([c.h for c in comps], dtype=np.float64)
    big = hs[hs > 0.4 * scale]
    char_h = float(np.median(big)) if len(big) else float(hs.max())
    if char_h < 6:
        return []
    band_y0 = system.lyric_rows[0][0]
    band_y1 = system.lyric_rows[-1][1]
    marks: List[Tuple[float, int]] = []
    for c in comps:
        if c.h > 0.62 * char_h:
            continue
        if c.w > max(3, 0.85 * char_h):
            continue
        if c.h < 0.3 * char_h:            # too small: noise / dot of an i-like glyph
            continue
        if c.cy < band_y0 + 0.35 * (band_y1 - band_y0):
            continue                       # must sit low in the lyric band
        marks.append((c.cx, c.h))
    marks.sort()
    merged: List[Tuple[float, int]] = []
    for x, h in marks:
        if merged and abs(x - merged[-1][0]) < 0.7 * scale:
            continue
        merged.append((x, h))
    return merged


def bar_positions(page: Page, system: System, scale: float) -> List[float]:
    """x of long vertical rules inside the music row (barlines)."""
    if page.rules is None:
        return []
    y0, y1 = system.y0, min(system.y1, page.rules.shape[0])
    if y1 - y0 < 4:
        return []
    col = (page.rules[y0:y1] > 0).sum(axis=0)
    thr = 0.35 * (y1 - y0)
    xs: List[float] = []
    run: List[int] = []
    for x, v in enumerate(col):
        if v >= thr:
            run.append(x)
        elif run:
            xs.append(float(np.mean(run)))
            run = []
    if run:
        xs.append(float(np.mean(run)))
    return xs


def _cut(n_notes: int, cut_after: Sequence[int]) -> List[Tuple[int, int]]:
    bounds = [0]
    for c in cut_after:
        if c + 1 > bounds[-1] and c + 1 < n_notes:
            bounds.append(c + 1)
    bounds.append(n_notes)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def phrases_from_marks(notes, marks: Sequence[Tuple[float, int]]) -> List[Tuple[int, int]]:
    if not notes:
        return []
    cuts = []
    for x, _h in marks:
        idx = -1
        for i, n in enumerate(notes):
            if n.x <= x:
                idx = i
        if idx >= 0:
            cuts.append(idx)
    return _cut(len(notes), cuts)


def phrases_by_bars(notes, bars: Sequence[float], bars_per_phrase: int = 4) -> List[Tuple[int, int]]:
    if not notes:
        return []
    if not bars:
        return []
    cuts = []
    for k in range(bars_per_phrase, len(bars), bars_per_phrase):
        x = bars[k - 1]
        idx = -1
        for i, n in enumerate(notes):
            if n.x <= x:
                idx = i
        if idx >= 0:
            cuts.append(idx)
    return _cut(len(notes), cuts)


def split_notes(page: Page, system: System, scale: float, bars_per_phrase: int = 4):
    notes = system.notes
    if not notes:
        return [], "空行"
    marks = punctuation_marks(page, system, scale)
    if marks:
        ph = phrases_from_marks(notes, marks)
        if ph:
            return ph, "歌词标点"
    bars = bar_positions(page, system, scale)
    if bars:
        ph = phrases_by_bars(notes, bars, bars_per_phrase)
        if ph:
            return ph, "小节线"
    return [(0, len(notes))], "整行"
