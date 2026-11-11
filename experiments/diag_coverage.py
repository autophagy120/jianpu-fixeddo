"""诊断：一条谱行里到底有多少个数字，我认出来多少，漏的都长什么样。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

from jp import core
from jp.glyphnet import GlyphNet


def digit_like(comps, scale):
    """候选数字：高度在主字高附近、宽高比像数字。这是"应该被识别"的分母。"""
    out = []
    for c in comps:
        if 0.55 * scale <= c.h <= 1.45 * scale and 0.18 <= c.aspect <= 1.15 and c.area >= 20:
            out.append(c)
    return out


def main(path):
    net = GlyphNet()
    g = core.load_gray(path)
    pg = core.analyse(g, net, path=path)
    k = pg.k or 1.0
    print(f"scale={pg.scale:.0f} k={k:.2f} rows={len(pg.rows)} systems={len(pg.systems)}")
    tot_cand = tot_got = 0
    for si, s in enumerate(pg.systems):
        row = pg.rows[[r for r in pg.rows if r.y0 == s.y0][0]] if False else None
        cand = digit_like([c for c in pg.comps if s.y0 <= c.cy < s.y1 and c.h > 0.55 * pg.scale],
                          pg.scale)
        got = len(s.notes)
        tot_cand += len(cand)
        tot_got += got
        miss = max(0, len(cand) - got)
        print(f"  行{si:2d} y={s.y0:5.0f}-{s.y1:5.0f} 候选={len(cand):3d} 识别={got:3d} 漏={miss:3d}")
    print(f"合计 候选 {tot_cand}  识别 {tot_got}  覆盖率 {tot_got / max(1, tot_cand):.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
