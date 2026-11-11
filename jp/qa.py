"""Visual QA: draw what the recogniser thinks onto the original page."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core, emit
from . import music as M


def annotate(bgr: np.ndarray, page: core.Page, key_pc: Optional[int] = None) -> np.ndarray:
    k = page.k or 1.0
    rgb = Image.fromarray(bgr[:, :, ::-1])
    d = ImageDraw.Draw(rgb)
    W, H = rgb.size
    f = emit._font(max(10, int(0.016 * W)))

    for si, sysn in enumerate(page.systems):
        y0 = int(sysn.y0 / k)
        d.line([(0, y0), (W, y0)], fill=(120, 180, 255), width=1)
        for n in sysn.notes:
            x0, y0_, x1, y1 = [int(round(v / k)) for v in n.box]
            col = (0, 170, 0) if n.conf >= 0.8 else ((230, 150, 0) if n.conf >= 0.6 else (220, 30, 30))
            d.rectangle([x0, y0_, x1, y1], outline=col, width=1)
            txt = n.digit
            if n.acc:
                txt = n.acc + txt
            if n.octave > 0:
                txt += "^" * n.octave
            elif n.octave < 0:
                txt += "v" * (-n.octave)
            d.text((x0, max(0, y0_ - int(0.012 * H))), txt, font=f, fill=col)
    return np.array(rgb)[:, :, ::-1].copy()


def side_by_side(a: np.ndarray, b: np.ndarray, path: str) -> None:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    H = max(ha, hb)
    canvas = np.full((H, wa + wb + 12, 3), 255, np.uint8)
    canvas[:ha, :wa] = a
    canvas[:hb, wa + 12:] = b
    core.imwrite(path, canvas)
