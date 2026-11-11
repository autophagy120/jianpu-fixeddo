"""三月雨最终产出 v2：插入受墨迹约束 + 未识别字形显式标出。"""
import json
import os
import sys

sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core, emit, propagate, score, vlmfill2
from jp import music as M
from jp.glyphnet import GlyphNet

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
REF = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小] (原调#G-转调C).jpg"
OUT = r"out\三月雨_vlm2"
CACHE = r"out\三月雨_vlm\lines.json"
KEY, TARGET = 8, 0


def row_components(page, row):
    """该音乐行里字形级的组件（原始工作分辨率）。"""
    out = []
    for c0 in row.comps:
        for c in core.split_wide_component(page.clean, c0, page.scale):
            if c.h >= 0.45 * page.scale and c.area >= 8 and 0.14 <= c.aspect <= 1.35:
                out.append(c)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    net = GlyphNet()
    page = core.analyse(core.load_gray(P), net, path=P)
    bgr = core.load_bgr(P)
    k = page.k or 1.0
    cache = json.load(open(CACHE, encoding="utf8")) if os.path.exists(CACHE) else {}
    rows = [r for r in page.rows if r.kind == "music"]

    dropped = 0
    for si, row in enumerate(rows):
        toks = cache.get(str(si), [])
        sysn = next((s for s in page.systems if abs(s.y0 - row.y0) < 2), None)
        if sysn is None or not toks:
            continue
        comps = row_components(page, row)
        vlmfill2.apply_tiles(page, sysn, [vlmfill2.TileRecord(0, bgr.shape[1], toks, 1.0)],
                             components=comps)

    notes = [n for s in page.systems for n in s.notes]
    c0 = sum(1 for n in notes if getattr(n, "confirmed", False))
    propagate.propagate(page, min_iou=0.82, margin=0.04)
    c1 = sum(1 for n in notes if getattr(n, "confirmed", False))
    p1 = sum(1 for n in notes if getattr(n, "propagated", False))
    print(f"确认 {c1} + 自举 {p1} / 共 {len(notes)} = {round((c1+p1)/max(1,len(notes))*100)}%")

    # ---- 未识别字形：该行字形级组件中没被任何音符覆盖的
    orphans = []
    for sysn in page.systems:
        if not sysn.notes:
            continue
        boxes = [(min(n.box[0], n.box[2]), max(n.box[0], n.box[2])) for n in sysn.notes]
        for c in sysn.notes[0].__class__ and [] or []:
            pass
    for row in rows:
        sysn = next((s for s in page.systems if abs(s.y0 - row.y0) < 2), None)
        if sysn is None or not sysn.notes:
            continue
        boxes = [(min(n.box[0], n.box[2]), max(n.box[0], n.box[2])) for n in sysn.notes]
        for c in row_components(page, row):
            cx = c.x + c.w / 2.0
            if any(a - 2 <= cx <= b + 2 for a, b in boxes):
                continue
            orphans.append((int(c.x / k), int(c.y / k),
                            int((c.x + c.w) / k), int((c.y + c.h) / k)))
    print(f"未识别字形 {len(orphans)} 个（橙色虚框 + ?）")

    # ---- 图
    newnotes = []
    for sysn in page.systems:
        row_ = []
        for n in sysn.notes:
            ok = getattr(n, "confirmed", False) or getattr(n, "propagated", False)
            if n.digit == "0":
                row_.append(core.Note(digit="0", box=n.box, x=n.x))
                continue
            deg, acc, octv = M.transpose_movable(int(n.digit), n.acc, n.octave, KEY, TARGET,
                                                 octave_mode="absolute")
            nn = core.Note(digit=str(deg), acc=acc, octave=octv, box=n.box, x=n.x,
                           acc_box=n.acc_box, dot_boxes=list(n.dot_boxes))
            nn.confirmed = ok
            row_.append(nn)
        newnotes.append(row_)
    img = emit.redraw_notes(bgr, page, newnotes,
                            emit.ImageOptions(footer=True,
                                              footer_text="原调 bA -> 转调 C   红=已确认  橙=未识别/待核对"),
                            orphans=orphans)
    core.imwrite(os.path.join(OUT, "transposed.jpg"), img)
    print("图:", os.path.join(OUT, "transposed.jpg"))

    if os.path.exists(REF):
        ref = core.load_bgr(REF)
        H = 1500
        def fit(im):
            return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_AREA)
        a, b, c = fit(bgr), fit(img), fit(ref)
        gap = np.full((H, 14, 3), 90, np.uint8)
        core.imwrite(os.path.join(OUT, "compare3.jpg"), np.hstack([a, gap, b, gap, c]))
        print("对照:", os.path.join(OUT, "compare3.jpg"))


if __name__ == "__main__":
    main()
