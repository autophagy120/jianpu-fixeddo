"""量一下三档「识别强度」在纯蓝上的效果：音符数 / 橙框数 / 耗时。"""
import os
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.merge_octave as MO
from jp.glyphnet import GlyphNet
import jp.emit_fixed as EF

it = [x for x in L.load().values() if (x.get("title") or "") == "纯蓝"][0]
txt = open(it["ocr_txt"], encoding="utf8").read()
net = GlyphNet()

import time
for st in (0, 1, 2):
    t0 = time.time()
    lines, pairs, stat, page = MO.merge_page(it["image"], txt, net, strength=st)
    unc = EF.find_uncovered(page, pairs)
    print("强度 %d：音符 %3d  橙框 %3d  补漏 boxes=%s geom=%s  agree=%.2f  耗时 %.1fs"
          % (st, len(pairs), len(unc),
             stat.get("filled_from_boxes"), stat.get("filled_geom"),
             stat.get("agree_rate", 0), time.time() - t0))
