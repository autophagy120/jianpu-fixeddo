"""三月雨：整行一次性 VLM 读取（1 次调用/行），缓存后可离线重放。"""
import json
import os
import sys

sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core, vlm, vlmfill2
from jp.glyphnet import GlyphNet

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
OUT = r"out\三月雨_vlm"
CACHE = os.path.join(OUT, "lines.json")
SCALE = 3.0
PAD = 6


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = json.load(open(CACHE, encoding="utf8")) if os.path.exists(CACHE) else {}
    from jp.glyphnet import GlyphNet as GN
    net = GN()
    page = core.analyse(core.load_gray(P), net, path=P)
    bgr = core.load_bgr(P)
    k = page.k or 1.0
    rows = [r for r in page.rows if r.kind == "music"]
    print(f"音乐行 {len(rows)}", flush=True)

    tc = ti = tu = 0
    for si, row in enumerate(rows):
        y0 = max(0, int(row.y0 / k) - PAD)
        y1 = min(bgr.shape[0], int(row.y1 / k) + PAD)
        key = str(si)
        if key in cache:
            toks = cache[key]
            print(f"  行{si:2d}: 缓存 {len(toks)} 音", flush=True)
        else:
            big = cv2.resize(bgr[y0:y1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
            toks = vlm.read_note_string(big, temperature=0.3, max_tokens=16000)
            cache[key] = toks
            json.dump(cache, open(CACHE, "w", encoding="utf8"), ensure_ascii=False)
            print(f"  行{si:2d}: y={y0}-{y1} -> {len(toks)} 音  {' '.join(toks[:14])}", flush=True)
        # 落到音符上
        sysn = next((s for s in page.systems if abs(s.y0 - row.y0) < 2), None)
        if sysn is None or not toks:
            continue
        tr = vlmfill2.TileRecord(0, bgr.shape[1], toks, 1.0)
        res = vlmfill2.apply_tiles(page, sysn, [tr])
        tc += res.confirmed
        ti += res.inserted
        tu += res.uncertain
    print(f"\n合计 确认 {tc} 插入 {ti} 待核 {tu}", flush=True)


if __name__ == "__main__":
    main()
