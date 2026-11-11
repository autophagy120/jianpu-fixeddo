"""三月雨：逐行 VLM 读取（片上收敛闸门 + 比例对齐 + 插入漏检），缓存每片以便离线重放。"""
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np

from jp import core, vlmfill2
from jp.glyphnet import GlyphNet

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
OUT = r"out\三月雨_vlm"
CACHE = os.path.join(OUT, "tiles.json")
SCALES = (6.0,)
SAMPLES = 2


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = json.load(open(CACHE, encoding="utf8")) if os.path.exists(CACHE) else {}
    net = GlyphNet()
    page = core.analyse(core.load_gray(P), net, path=P)
    bgr = core.load_bgr(P)
    k = page.k or 1.0
    print(f"行数 {len(page.systems)}", flush=True)

    tc = ti = tu = 0
    for si, sysn in enumerate(page.systems):
        if not sysn.notes:
            continue
        row = None
        for r in page.rows:
            if r.kind == "music" and abs(r.y0 - sysn.y0) < 2:
                row = r
                break
        y0 = int((row.y0 if row else sysn.y0) / k) - 6
        y1 = int((row.y1 if row else sysn.y0 + 40) / k) + 6
        key = str(si)
        if key in cache:
            tiles = [vlmfill2.TileRecord(*t) for t in cache[key]]
            res = vlmfill2.apply_tiles(page, sysn, tiles, dbg=True)
            print(f"  行{si:2d}: 缓存重放 确认{res.confirmed} 插入{res.inserted} 待核{res.uncertain}",
                  flush=True)
        else:
            res = vlmfill2.read_system(page, sysn, bgr, y0, y1, dbg=True,
                                       scales=SCALES, samples=SAMPLES, step_ratio=0.85)
            cache[key] = [t.as_list() for t in res.tiles]
            json.dump(cache, open(CACHE, "w", encoding="utf8"), ensure_ascii=False, indent=1)
            print(f"  行{si:2d}: 确认{res.confirmed} 插入{res.inserted} 待核{res.uncertain} "
                  f"(覆盖 {res.coverage:.0%})", flush=True)
        tc += res.confirmed
        ti += res.inserted
        tu += res.uncertain
    print(f"\n合计 确认 {tc} / 插入 {ti} / 待核 {tu}", flush=True)


if __name__ == "__main__":
    main()
