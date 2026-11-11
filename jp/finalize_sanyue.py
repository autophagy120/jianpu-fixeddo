"""三月雨最终产出：缓存重放 VLM -> 自举传播 -> 出图(红/橙) + 文本谱。"""
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np

from jp import core, emit, propagate, score, vlmfill2
from jp import music as M
from jp.glyphnet import GlyphNet

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
OUT = r"out\三月雨_vlm"
CACHE = os.path.join(OUT, "tiles.json")
KEY, TARGET = 8, 0


def main():
    net = GlyphNet()
    page = core.analyse(core.load_gray(P), net, path=P)
    bgr = core.load_bgr(P)
    k = page.k or 1.0
    cache = json.load(open(CACHE, encoding="utf8")) if os.path.exists(CACHE) else {}

    for si, sysn in enumerate(page.systems):
        tiles = [vlmfill2.TileRecord(*t) for t in cache.get(str(si), [])]
        vlmfill2.apply_tiles(page, sysn, tiles)

    stat = propagate.propagate(page, dbg=True)
    print(f"\n自举: 模板 {stat['templates']} 个，补认 {stat['filled']}，仍待核 {stat['left']}")

    notes = [n for s in page.systems for n in s.notes]
    conf = sum(1 for n in notes if getattr(n, "confirmed", False))
    prop = sum(1 for n in notes if getattr(n, "propagated", False))
    other = len(notes) - conf - prop
    print(f"音符总计 {len(notes)}：VLM确认 {conf} + 自举 {prop} + 待核 {other} "
          f"→ 自动 {round((conf + prop) / max(1, len(notes)) * 100)}%")

    # ---- 转调图
    newnotes = []
    for sysn in page.systems:
        row = []
        for n in sysn.notes:
            if n.digit == "0":
                row.append(core.Note(digit="0", box=n.box, x=n.x))
                continue
            deg, acc, octv = M.transpose_movable(int(n.digit), n.acc, n.octave, KEY, TARGET,
                                                 octave_mode="absolute")
            nn = core.Note(digit=str(deg), acc=acc, octave=octv, box=n.box, x=n.x,
                           acc_box=n.acc_box, dot_boxes=list(n.dot_boxes))
            nn.confirmed = getattr(n, "confirmed", False) or getattr(n, "propagated", False)
            row.append(nn)
        newnotes.append(row)
    img = emit.redraw_notes(bgr, page, newnotes,
                            emit.ImageOptions(footer=True,
                                              footer_text="原调 bA -> 转调 C   红=已确认  橙=待核对"))
    core.imwrite(os.path.join(OUT, "transposed.jpg"), img)
    print("图:", os.path.join(OUT, "transposed.jpg"))

    # ---- 文本谱
    harm = M.HarmonicaOptions()
    topts = emit.TextOptions(style="cn", harmonica=True)
    pitches = [M.note_to_abs(int(n.digit), n.acc, n.octave, KEY)
               for s in page.systems for n in s.notes if n.digit != "0"]
    ref_pc, reason = M.pick_reference(pitches, 3)
    page_lines, bias = score.build_page(page, KEY, ref_pc, harm, topts, None)
    body = []
    for si, lines in page_lines:
        for ln in lines:
            body.append(" ".join(ln.tokens))
        body.append("")
    head = ["# 三月雨（VLM 确认 + 自举补全）",
            f"# 原调 1=bA   固定调 1={M.key_name(ref_pc)}   {reason}",
            f"# 自动识别 {conf + prop}/{len(notes)} 个音符，其余需人工核对"]
    with open(os.path.join(OUT, "score.txt"), "w", encoding="utf8") as fh:
        fh.write("\n".join(head) + "\n\n" + "\n".join(body))
    print("文本:", os.path.join(OUT, "score.txt"))


if __name__ == "__main__":
    main()
