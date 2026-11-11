"""三月雨：出转调图（红=已确认 / 橙=待核对）与固定调文本谱（待核对加 ?）。"""
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np

from jp import core, emit, score, vlmfill2
from jp import music as M
from jp.glyphnet import GlyphNet

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
OUT = r"out\三月雨_vlm"
KEY, TARGET = 8, 0
CACHE = os.path.join(OUT, "_lines.json")


def main():
    os.makedirs(OUT, exist_ok=True)
    net = GlyphNet()
    page = core.analyse(core.load_gray(P), net, path=P)
    bgr = core.load_bgr(P)
    k = page.k or 1.0

    # 用缓存（run_sanyue 已经把结果写进 _lines.json；这里重放 VLM 结果太贵，
    # 所以直接把缓存里的 token 重新落到音符上）
    cache = json.load(open(CACHE, encoding="utf8")) if os.path.exists(CACHE) else {}
    total_c = total_u = 0
    for si, sysn in enumerate(page.systems):
        key = str(si)
        if key not in cache:
            continue
        toks = cache[key].get("line_tokens", [])
        if not toks:
            continue
        notes = sorted([n for n in sysn.notes if n.digit != "0"], key=lambda n: n.x)
        # 整行一对一（只有数量完全相等才敢配）
        if len(toks) == len(notes):
            for n, t in zip(notes, toks):
                parsed = vlmfill2.parse_token(t)
                if not parsed:
                    continue
                acc, d, o = parsed
                n.digit = d
                if o:
                    n.octave = o
                if acc in "#b":
                    n.acc = acc
                n.conf = 0.95
                n.confirmed = True
        else:
            print(f"  行{si}: 行级数量不等（框 {len(notes)} vs token {len(toks)}），整行保留原判")
    for s in page.systems:
        for n in s.notes:
            if getattr(n, "confirmed", False):
                total_c += 1
            else:
                total_u += 1
    print(f"确认 {total_c} / 待核 {total_u}")

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
            nn.confirmed = getattr(n, "confirmed", False)
            row.append(nn)
        newnotes.append(row)
    img = emit.redraw_notes(bgr, page, newnotes,
                            emit.ImageOptions(footer=True, footer_text="原调 bA -> 转调 C（红=已确认 橙=待核对）"))
    core.imwrite(os.path.join(OUT, "transposed.jpg"), img)
    print("图:", os.path.join(OUT, "transposed.jpg"))

    # ---- 文本谱（待核对加 ?）
    harm = M.HarmonicaOptions()
    topts = emit.TextOptions(style="cn", harmonica=True, annotate=False)
    ref_pc, reason = M.pick_reference([M.note_to_abs(int(n.digit), n.acc, n.octave, KEY)
                                       for s in page.systems for n in s.notes if n.digit != "0"], 3)
    page_lines, bias = score.build_page(page, KEY, ref_pc, harm, topts, None)
    body = score.render_page(page_lines, mode="none")
    head = [f"# 三月雨（VLM 确认版）",
            f"# 原调 1=bA   固定调 1={M.key_name(ref_pc)}   {reason}",
            f"# 已确认 {total_c} 个音符 / 待核对 {total_u} 个（文本里以 ? 结尾）"]
    with open(os.path.join(OUT, "score.txt"), "w", encoding="utf8") as fh:
        fh.write("\n".join(head) + "\n\n" + "\n".join(body))
    print("文本:", os.path.join(OUT, "score.txt"))


if __name__ == "__main__":
    main()
