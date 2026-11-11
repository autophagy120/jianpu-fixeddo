"""用 Xiaomi-OCR-0 的结果生成固定调文本谱（含歌词分句）。本地、零成本。"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, ".")
import re

from jp import lyrics as LY
from jp import music as M
from jp import xiaomi_ocr as XO


def pair_lines(text: str) -> List[Tuple[XO.OcrLine, str]]:
    """把谱行和它下面紧跟的歌词行配对。"""
    raw_lines = text.splitlines()
    parsed = [XO.parse_line(t) for t in raw_lines]
    pairs: List[Tuple[XO.OcrLine, str]] = []
    for i, ln in enumerate(parsed):
        if not ln.is_music:
            continue
        lyric = ""
        for j in range(i + 1, min(i + 4, len(parsed))):
            cand = parsed[j]
            if cand.is_music:
                break
            if re.search(r"[\u4e00-\u9fff]", cand.raw):
                lyric = cand.raw.strip()
                break
        pairs.append((ln, lyric))
    return pairs


def fixed_do_tokens(notes: List[XO.OcrNote], key_pc: int, ref_pc: int,
                    harm: M.HarmonicaOptions) -> List[str]:
    idx = [i for i, n in enumerate(notes) if n.digit != "0"]
    toks = ["0"] * len(notes)
    if not idx:
        return toks
    pitches = [M.note_to_abs(int(notes[i].digit), notes[i].acc, notes[i].octave, key_pc) for i in idx]
    # 八度基准固定为 0：绝对音高，C4 为不分八度的基准。
    # （之前用"中位音区居中"，会把整行往下挪一个八度，看起来像高低音读错了）
    bias = 0
    rendered = M.render_fixed_do(pitches, ref_pc, harm, "cn", harmonica=True, octave_bias=bias)
    for t, i in zip(rendered, idx):
        toks[i] = t
    return toks


def main():
    img = r"E:\jianpu-env\ocr\test\page.png"
    txt = XO.ocr_image(img)
    open(r"out\三月雨_ocr.txt", "w", encoding="utf8").write(txt)

    pairs = pair_lines(txt)
    notes_all = [n for ln, _ly in pairs for n in ln.notes if n.digit != "0"]
    print(f"谱行 {len(pairs)} 条，音符 {len(notes_all)} 个")

    key_pc = 8           # 1=bA
    harm = M.HarmonicaOptions()
    pitches = [M.note_to_abs(int(n.digit), n.acc, n.octave, key_pc) for n in notes_all]
    ref_pc, reason = M.pick_reference(pitches, 3)

    out = ["# 三月雨 —— 固定调文本谱（Xiaomi-OCR-0 本地识别）",
           f"# 原调 1=bA   固定调 1={M.key_name(ref_pc)}   {reason}",
           f"# 本地 OCR，零 API 调用", ""]
    for li, (ln, lyric) in enumerate(pairs):
        toks = fixed_do_tokens(ln.notes, key_pc, ref_pc, harm)
        chars = [c for c in lyric if re.match(r"[\u4e00-\u9fff]", c)]
        # 按歌词标点分句
        phrases = LY.split_phrases(lyric) if lyric else []
        if chars and phrases:
            # 逐字对齐（按音组），再按标点切句
            spans = LY.align_chars(len(toks), chars, max_per_char=4)
            cuts, k = [], 0
            for ch in lyric:
                if re.match(r"[\u4e00-\u9fff]", ch):
                    k += 1
                elif LY.is_punct(ch) and k > 0 and k < len(chars):
                    cuts.append(k - 1)
            bounds = [0]
            for c in cuts:
                if c + 1 > bounds[-1] and c + 1 < len(chars):
                    bounds.append(c + 1)
            bounds.append(len(chars))
            for bi in range(len(bounds) - 1):
                c0, c1 = bounds[bi], bounds[bi + 1]
                if c1 <= c0:
                    continue
                a = spans[c0][0]
                b = spans[c1 - 1][1]
                out += LY.render("align", toks[a:b], chars[c0:c1],
                                 [(max(0, spans[x][0] - a), min(b - a, spans[x][1] - a))
                                  for x in range(c0, c1)])
        else:
            out.append(" ".join(toks))
        out.append("")

    path = r"out\三月雨_textscore.txt"
    open(path, "w", encoding="utf8").write("\n".join(out))
    print("写出:", path)
    print("\n".join(out[:14]))


if __name__ == "__main__":
    main()
