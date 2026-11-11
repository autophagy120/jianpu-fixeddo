"""三月雨最终产出：整行 VLM 结果 -> 自举传播 -> 转调图(红/橙) + 文本谱 + 与 easytr 对照。"""
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np

from jp import core, emit, propagate, score, vlmfill2
from jp import music as M
from jp.glyphnet import GlyphNet

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
REF = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小] (原调#G-转调C).jpg"
OUT = r"out\三月雨_vlm"
CACHE = os.path.join(OUT, "lines.json")
KEY, TARGET = 8, 0


def main():
    net = GlyphNet()
    page = core.analyse(core.load_gray(P), net, path=P)
    bgr = core.load_bgr(P)
    k = page.k or 1.0
    cache = json.load(open(CACHE, encoding="utf8"))
    rows = [r for r in page.rows if r.kind == "music"]

    for si, row in enumerate(rows):
        toks = cache.get(str(si), [])
        sysn = next((s for s in page.systems if abs(s.y0 - row.y0) < 2), None)
        if sysn is None or not toks:
            continue
        tr = vlmfill2.TileRecord(0, bgr.shape[1], toks, 1.0)
        vlmfill2.apply_tiles(page, sysn, [tr])

    notes = [n for s in page.systems for n in s.notes]
    c0 = sum(1 for n in notes if getattr(n, "confirmed", False))
    print(f"VLM 确认 {c0}/{len(notes)}")
    stat = propagate.propagate(page, min_iou=0.82, margin=0.04)
    c1 = sum(1 for n in notes if getattr(n, "confirmed", False))
    p1 = sum(1 for n in notes if getattr(n, "propagated", False))
    left = len(notes) - c1 - p1
    print(f"自举补 {p1} 个 -> 自动 {c1 + p1}/{len(notes)} = {round((c1+p1)/max(1,len(notes))*100)}%")

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
                                              footer_text="原调 bA -> 转调 C   红=自动确认  橙=待核对"))
    core.imwrite(os.path.join(OUT, "transposed.jpg"), img)

    # ---- 对照图（原图 | 我的 | easytr）
    if os.path.exists(REF):
        ref = core.load_bgr(REF)
        H = 1500
        def fit(im):
            import cv2
            return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_AREA)
        import cv2
        a, b, c = fit(bgr), fit(img), fit(ref)
        gap = np.full((H, 14, 3), 90, np.uint8)
        core.imwrite(os.path.join(OUT, "compare3.jpg"), np.hstack([a, gap, b, gap, c]))
        print("对照图:", os.path.join(OUT, "compare3.jpg"))

    # ---- 文本谱
    harm = M.HarmonicaOptions()
    topts = emit.TextOptions(style="cn", harmonica=True)
    pitches = [M.note_to_abs(int(n.digit), n.acc, n.octave, KEY)
               for n in notes if n.digit != "0"]
    ref_pc, reason = M.pick_reference(pitches, 3)
    page_lines, bias = score.build_page(page, KEY, ref_pc, harm, topts, None)
    body = []
    for si, lines in page_lines:
        for ln in lines:
            body.append(" ".join(ln.tokens))
        body.append("")
    head = ["# 三月雨（VLM 整行读取 + 自举补全）",
            f"# 原调 1=bA   固定调 1={M.key_name(ref_pc)}   {reason}",
            f"# 自动识别 {c1 + p1}/{len(notes)}（VLM {c1} + 自举 {p1}），待核对 {left}"]
    open(os.path.join(OUT, "score.txt"), "w", encoding="utf8").write("\n".join(head) + "\n\n" + "\n".join(body))
    print("文本:", os.path.join(OUT, "score.txt"))


if __name__ == "__main__":
    main()
