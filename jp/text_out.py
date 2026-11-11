"""输出文本谱：原谱(首调) / 固定调 / 绝对音名 / 逐音歌词。

歌词逐音对齐是这个模块的核心难点，最终方案是**三级锚定**：

1. **谱行级锚定**：几何谱行数与 OCR 谱行数并不相等（为了你 17 vs 15），
   任何"按序号/按比例"的映射都会整体串行。改用**音符数做锚点做序列对齐**
   （编辑距离 DP），把几何谱行和 OCR 谱行一一配对。
2. **歌词行级锚定**：配对好的谱行，其后面跟的几何歌词行与 OCR 歌词文本行
   再按顺序配对 —— 到这层两边数量基本一致。
3. **字符级锚定**：文字用 Xiaomi-OCR（完整准确），水平位置用几何
   （歌词墨块按间距聚成字，一个不漏），行内按顺序 zip 后按 x 挂到最近的音符。
   无词位置留空 `_`，前奏/间奏因此能占位 —— 排版可用。

`--target` 等于原调时是**恒等输出**，数字应与原谱逐音一致，用于核对识别。

用法：
    python -m jp.text_out --song 三月雨
    python -m jp.text_out --song 为了你 --target C
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import merge_octave as MO
from . import music as M
from . import xiaomi_ocr as XO

SONGS = {
    "三月雨": (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg",
             r"out\batch\_ocr\三月雨__50b751ebbe319 [原始大小].txt", "bA"),
    "为了你": (r"work\为了你唱下去.jpg", r"work\为了你唱下去.ocr.txt", "G"),
}

_PC = ["C", "#C", "D", "#D", "E", "F", "#F", "G", "#G", "A", "#A", "B"]


def _pname(p) -> str:
    return f"{_PC[int(p.pc) % 12]}{int(p.octave)}"


def _movable(notes) -> str:
    out = []
    for n in notes:
        t = ("#" if n.acc == "#" else ("b" if n.acc == "b" else "")) + n.digit
        if n.octave > 0:
            t += "\u0307" * n.octave
        elif n.octave < 0:
            t += "\u0323" * (-n.octave)
        if n.aug_dot:
            t += "·"
        out.append(t)
    return " ".join(out)


# ---------------------------------------------------------------- 序列对齐

_GAP = -1.5
_SUB = 4.0


def align_by_counts(a: Sequence[int], b: Sequence[int]) -> List[Optional[int]]:
    """按**元素数值接近程度**做单调对齐，返回 b 中每个元素对应的 a 下标。

    这里用于"几何谱行 ↔ OCR 谱行"：数值是各自的音符数。相等最好，
    差几个也能配上，多出来的一侧走 gap。这样即使两边行数不等也不会整体串行。
    """
    m, n = len(b), len(a)
    if m == 0 or n == 0:
        return [None] * m
    NEG = -1e18
    dp = np.full((m + 1, n + 1), NEG)
    bt = np.zeros((m + 1, n + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] + _GAP
        bt[0, j] = 2
    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] + _GAP
        bt[i, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diff = abs(b[i - 1] - a[j - 1])
            best = dp[i - 1, j - 1] + _SUB - min(3.0, diff)
            mv = 0
            v = dp[i - 1, j] + _GAP
            if v > best:
                best, mv = v, 1
            v = dp[i, j - 1] + _GAP
            if best < v:
                best, mv = v, 2
            dp[i, j] = best
            bt[i, j] = mv
    out: List[Optional[int]] = [None] * m
    i, j = m, n
    while i > 0 or j > 0:
        mv = bt[i, j]
        if i > 0 and j > 0 and mv == 0:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and mv == 1:
            i -= 1
        elif j > 0 and mv == 2:
            j -= 1
        else:
            i, j = i - 1, j - 1
    return out


# ---------------------------------------------------------------- 歌词

def _char_spans(row, scale: float) -> List[Tuple[float, float]]:
    """歌词行的墨块按 x 间距聚成"字"，返回 [(x0, x1)]。

    间距阈值必须用**歌词行自己的字高**，不能用页级字高：
    页级字高被谱行数字撑大（三月雨 38px），0.42×38≈16px 的阈值会把相邻歌词字
    合并成一个"位置"，结果多个字挤到同一个音上（`了你`、`荒芜`、`细雨`）。
    """
    cs = [c for c in row.comps if c.h >= 0.22 * scale and c.area >= 6]
    if not cs:
        return []
    lh = float(np.median([c.h for c in cs]))
    gap_th = max(2.0, 0.30 * lh)
    cs.sort(key=lambda c: c.x)
    groups, cur = [], []
    for c in cs:
        if cur and (c.x - (cur[-1].x + cur[-1].w)) > gap_th:
            groups.append(cur)
            cur = [c]
        else:
            cur.append(c)
    if cur:
        groups.append(cur)
    return [(min(c.x for c in g), max(c.x + c.w for c in g)) for g in groups]


def _is_lyric_text(t: str) -> bool:
    t = t.strip()
    if not t or "=" in t:
        return False
    if any("a" <= c.lower() <= "z" for c in t):
        return False
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    return cjk >= 2 and cjk >= 0.5 * len(t.replace(" ", ""))


def _ocr_structure(txt: str):
    """返回 (谱行音符数列表, 每条谱行后面的歌词文本行列表)。"""
    lines = txt.splitlines()
    counts, lyrics = [], []
    cur: List[str] = []
    started = False
    for l in lines:
        p = XO.parse_line(l)
        if p.is_music:
            if started:
                lyrics.append(cur)
            started = True
            cur = []
            counts.append(len(p.notes))
        elif started and _is_lyric_text(l):
            cur.append(l.strip())
    if started:
        lyrics.append(cur)
    return counts, lyrics


def attach_lyrics(page, pairs, ocr_txt: str) -> Dict[int, List[str]]:
    """返回 {id(slot): [第1段词, 第2段词]}；无词给空串列表。"""
    if not pairs:
        return {}
    scale = float(page.scale)
    rows = page.rows

    # ---- 几何：谱行及其后紧跟的歌词行
    mus_rows, geo_lyr = [], []
    for i, r in enumerate(rows):
        if r.kind != "music":
            continue
        got, j = [], i + 1
        while j < len(rows) and rows[j].kind != "music":
            if rows[j].kind == "lyric":
                got.append(rows[j])
            j += 1
        mus_rows.append(r)
        geo_lyr.append(got)

    # ---- 每个音属于哪条几何谱行
    slot_row: Dict[int, object] = {}
    for s, _o in pairs:
        best, bestov = None, 0.0
        for r in mus_rows:
            ov = min(s.box[3], r.y1) - max(s.box[1], r.y0)
            if ov > bestov:
                best, bestov = r, ov
        slot_row[id(s)] = best
    geo_counts = [sum(1 for s, _o in pairs if slot_row.get(id(s)) is r) for r in mus_rows]

    # ---- OCR：谱行音符数 + 每行后歌词
    ocr_counts, ocr_lyr = _ocr_structure(ocr_txt)

    # ---- 一级锚定：几何谱行 -> OCR 谱行（按音符数 DP）
    g2o = align_by_counts(ocr_counts, geo_counts)

    out: Dict[int, List[str]] = {id(s): [] for s, _o in pairs}

    for gi, orow in enumerate(g2o):
        lrows = geo_lyr[gi]
        if not lrows:
            continue
        texts: List[str] = []
        if orow is not None and orow < len(ocr_lyr):
            texts = ocr_lyr[orow]
        # 二级锚定：歌词行 ↔ 歌词文本行（数量基本一致，直接按序）
        if not texts:
            continue
        inrow = sorted([s for s, o in pairs
                        if slot_row.get(id(s)) is mus_rows[gi] and str(o.digit) != "0"],
                       key=lambda s: s.x)
        if not inrow:
            continue
        for vi, lr in enumerate(lrows[:2]):
            txt = texts[vi] if vi < len(texts) else ""
            chars = [c for c in txt if not c.isspace()]
            spans = _char_spans(lr, scale)
            if not spans or not chars:
                continue
            # 三级锚定：字符 ↔ 水平位置。**按区间分配**，不是按比例取一个：
            # 位置比字少时（几个字共用一个墨块，如连音组），该位置要收下整段字；
            # 按"一个位置取一个字"会把中间的字直接丢掉（实测 `你`、`荒`、`字`、`写` 都丢过）。
        for vi, lr in enumerate(lrows[:2]):
            txt = texts[vi] if vi < len(texts) else ""
            chars = [c for c in txt if not c.isspace()]
            spans = _char_spans(lr, scale)
            if not chars:
                continue
            n_c, n_s = len(chars), len(spans)
            # 位置锚点怎么取：
            # - 墨块聚出来的"字位"数与字数接近时，用字位（它反映真实印刷位置，
            #   连音组会用同一个位置，和谱面一致）；
            # - 差得多时改用**音符自身 x 坐标等距取位**。歌词本来就印在音符正下方，
            #   一个是字对一个音；靠墨块边界反倒会把相邻字合并（`你唱`、`荒芜`）。
            if n_s and abs(n_s - n_c) <= max(1, int(0.25 * n_c)):
                for ci in range(min(n_c, n_s)):
                    cx = (spans[ci][0] + spans[ci][1]) / 2.0
                    k = min(range(len(inrow)), key=lambda z: abs(inrow[z].x - cx))
                    arr = out.setdefault(id(inrow[k]), [])
                    while len(arr) <= vi:
                        arr.append("")
                    arr[vi] = arr[vi] + chars[ci]
            else:
                # 字少音多：按比例**铺满整行**。早先只往前 N 个音上放，
                # 后面一截音全空（表现为一堆 `_`）。
                n_notes = len(inrow)
                for ci, ch in enumerate(chars):
                    k = int(round(ci * (n_notes - 1) / max(1, n_c - 1))) if n_c > 1 else 0
                    arr = out.setdefault(id(inrow[min(k, n_notes - 1)]), [])
                    while len(arr) <= vi:
                        arr.append("")
                    arr[vi] = arr[vi] + ch
    return out


def attach_lyrics(page, pairs, ocr_txt: str) -> Dict[int, List[str]]:
    """最终方案：**内容锚定 + 局部补字**。

    分工：
    - RapidOCR 给出歌词字的**坐标**（位置可靠，行内天然对齐，不会串行），
      但小字会漏检几个字；
    - Xiaomi-OCR 的歌词**文本完整准确**，但没有坐标。

    做法：对每条几何歌词行，用 RapidOCR 检出该行的字串 S；在 VLM 歌词行里找
    与 S 最相似的那一行 L（difflib 相似度，**按内容匹配而不是按行数/字数推测**，
    后者只要一行对不上就整块错位）。然后按 L 对齐补齐：S 里有的字用它的坐标，
    S 里缺的字按左右邻居的 x **插值**补位。这样既不串行、也不缺字。
    """
    import difflib
    from . import ocr_boxes as OB
    if not pairs:
        return {}
    scale = float(page.scale)
    rows = page.rows
    ly_rows = [r for r in rows if r.kind == "lyric"]
    out: Dict[int, List[str]] = {id(s): [] for s, _o in pairs}
    if not ly_rows:
        return out

    stem = os.path.basename(page.path)
    stem = "".join(c if c.isalnum() or "\u4e00" <= c <= "\u9fff" else "_" for c in stem)[:70]
    try:
        boxes, _s = OB.read_boxes(page.gray, use_cache=True,
                                  cache_path=os.path.join("out", "_ocrboxes", stem + ".json"))
        chars = OB.chars_from_boxes(boxes, digits_only=False)
    except Exception:
        return out

    vlm = [c for c in _lyric_text_lines(ocr_txt)]
    if not vlm:
        return out

    for lr in ly_rows:
        # 该歌词行内的检出字符（按 x 排序）
        inl = [c for c in chars
               if lr.y0 - 0.6 * scale <= c.cy <= lr.y1 + 0.6 * scale
               and not c.ch.isdigit() and not c.ch.isspace()
               and "\u4e00" <= c.ch or (c.ch in "，。、！？；：")]
        inl = [c for c in inl if c.ch.strip()]
        if not inl:
            continue
        inl.sort(key=lambda c: c.cx)
        s_det = "".join(c.ch for c in inl)
        xs_det = [c.cx for c in inl]
        if len(s_det) < 2:
            continue
        # 按内容找最相似的 VLM 歌词行
        best, bestr = None, 0.0
        for t in vlm:
            tt = "".join(ch for ch in t if not ch.isspace())
            r = difflib.SequenceMatcher(None, s_det, tt).ratio()
            if r > bestr:
                bestr, best = r, tt
        if best is None or bestr < 0.45:
            continue
        # 用 best 对齐，补齐 s_det 缺的字（按邻居 x 插值）
        seq: List[Tuple[str, float]] = []
        sm = difflib.SequenceMatcher(None, s_det, best)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    seq.append((s_det[i1 + k], xs_det[i1 + k]))
            elif tag == "replace":
                n = j2 - j1
                lo = xs_det[i1 - 1] if i1 > 0 else (xs_det[0] - 0.8 * scale)
                hi = xs_det[i1] if i1 < len(xs_det) else (xs_det[-1] + 0.8 * scale)
                for k in range(n):
                    x = lo + (hi - lo) * ((k + 1) / (n + 1))
                    seq.append((best[j1 + k], x))
            elif tag == "insert":
                n = j2 - j1
                lo = xs_det[i1 - 1] if i1 > 0 else (xs_det[0] - 0.8 * scale)
                hi = xs_det[i1] if i1 < len(xs_det) else (xs_det[-1] + 0.8 * scale)
                for k in range(n):
                    x = lo + (hi - lo) * ((k + 1) / (n + 1))
                    seq.append((best[j1 + k], x))
            # delete: S 里多出来的字（多为误检）直接丢弃
        if not seq:
            continue
        near = [s for s, _o in pairs
                if s.box[3] <= lr.y0 + 0.45 * scale and lr.y0 - s.box[3] <= 3.0 * scale]
        if not near:
            continue
        near.sort(key=lambda s: s.x)
        for ch, x in seq:
            k = min(range(len(near)), key=lambda z: abs(near[z].x - x))
            out.setdefault(id(near[k]), []).append(ch)
    return out


def _lyric_text_lines(ocr_txt: str) -> List[str]:
    lines = ocr_txt.splitlines()
    first = None
    for i, l in enumerate(lines):
        if XO.parse_line(l).is_music:
            first = i
            break
    out = []
    for i, l in enumerate(lines):
        t = l.strip()
        if not t or "=" in t:
            continue
        if first is not None and i <= first:
            continue
        if XO.parse_line(l).is_music:
            continue
        if any("a" <= c.lower() <= "z" for c in t):
            continue
        cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
        if cjk >= 2 and cjk >= 0.5 * len(t.replace(" ", "")):
            out.append(t)
    return out


def attach_lyrics(page, pairs, ocr_txt: str) -> Dict[int, List[str]]:
    """歌词逐音（采用版）：直接用 RapidOCR 的字符框逐字挂到最近的音。

    对比过的三种方案与实测覆盖率：
    - 顺序 zip（VLM 文本 + 几何位置）：行数不等时整行串位，否决；
    - 字数 DP 对齐：锚点弱，串行，否决；
    - 内容相似度 + 插值补字：best-line 选错时顺序会乱，覆盖率反而降到 45~57%，否决；
    - **RapidOCR 字符框（本版）**：字符自带坐标，行内天然对齐，不会串行，
      覆盖率最高。代价是对小字漏检少量字，表现为 `_`，但不影响排版对齐。
    """
    from . import ocr_boxes as OB
    if not pairs:
        return {}
    mh = float(np.median([max(1.0, s.box[3] - s.box[1]) for s, _o in pairs]))
    stem = os.path.basename(page.path)
    stem = "".join(c if c.isalnum() or "\u4e00" <= c <= "\u9fff" else "_" for c in stem)[:70]
    try:
        boxes, _s = OB.read_boxes(page.gray, use_cache=True,
                                  cache_path=os.path.join("out", "_ocrboxes", stem + ".json"))
        chars = OB.chars_from_boxes(boxes, digits_only=False)
    except Exception:
        return {}
    out: Dict[int, List[str]] = {id(s): [] for s, _o in pairs}
    for s, _o in pairs:
        x0, y0, x1, y1 = s.box
        cy0, cy1 = y1 + 0.15 * mh, y1 + 3.2 * mh
        cands = [c for c in chars
                 if cy0 <= c.cy <= cy1 and abs(c.cx - s.x) <= 0.62 * mh
                 and not c.ch.isdigit()]
        if not cands:
            continue
        cands.sort(key=lambda c: c.cy)
        verses, cur = [], []
        for c in cands:
            if cur and abs(c.cy - cur[0].cy) > 0.9 * mh:
                verses.append(cur)
                cur = [c]
            else:
                cur.append(c)
        if cur:
            verses.append(cur)
        res = []
        for v in verses[:2]:
            v.sort(key=lambda c: c.cx)
            res.append("".join(c.ch for c in v))
        out[id(s)] = res
    return out


def build(song: str, target: str = "") -> str:
    orig, ocr_path, key_name = SONGS[song]
    from .glyphnet import GlyphNet
    txt = MO.sanitize_ocr(open(ocr_path, encoding="utf8").read())
    lines, pairs, st, page = MO.merge_page(orig, txt, GlyphNet())
    key_pc = M.parse_key(key_name)
    ref_pc = M.parse_key(target) if target else key_pc
    harm = M.HarmonicaOptions()
    alln = [n for l in lines for n in l if n.digit != "0"]

    lyr = attach_lyrics(page, pairs, txt)
    note2slot = {id(o): s for s, o in pairs}

    def _l(n) -> str:
        s = note2slot.get(id(n))
        v = lyr.get(id(s)) if s is not None else None
        return (v[0] if v else "") or "_"

    head = [
        f"# {song} —— 文本谱",
        "# 数字: Xiaomi-OCR(RapidOCR 补漏)；高低八度点: 几何检测",
        f"# 原调 1={M.key_name(key_pc)}   输出 1={M.key_name(ref_pc)}"
        + ("   （与原调相同＝恒等输出，用于核对识别）" if ref_pc == key_pc else ""),
        "# 八度：绝对音高，以 C4 为基准，( ) 低八度、【 】 高八度",
        "# 歌词(逐音)：每列对应上方一个音；`_` = 该音无词（前奏/间奏/延音）",
        f"# 统计: 配对 {st['paired']} / 数字一致 {st['agree_rate']*100:.1f}% / "
        f"上点 {sum(1 for n in alln if n.octave > 0)} / "
        f"下点 {sum(1 for n in alln if n.octave < 0)}",
        "",
    ]
    body = []
    for i, notes in enumerate(lines):
        if not notes:
            continue
        idx = [k for k, n in enumerate(notes) if n.digit != "0"]
        ps = [M.note_to_abs(int(notes[k].digit), notes[k].acc,
                            int(notes[k].octave), key_pc) for k in idx]
        toks = ["0"] * len(notes)
        nms = ["休止"] * len(notes)
        if ps:
            rendered = M.render_fixed_do(ps, ref_pc, harm, "cn", harmonica=False, octave_bias=0)
            for t, nm, k in zip(rendered, [_pname(p) for p in ps], idx):
                toks[k] = t
                nms[k] = nm
        body.append(f"第{i+1:02d}行 原谱(首调)  {_movable(notes)}")
        body.append(f"        固定调1={M.key_name(ref_pc)}  " + " ".join(toks))
        body.append("        绝对音名            " + " ".join(nms))
        body.append("        歌词(逐音)          " + " ".join(_l(n) for n in notes))
        body.append("")
    return "\n".join(head + body)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="三月雨", choices=list(SONGS))
    ap.add_argument("--target", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    s = build(a.song, a.target)
    out = a.out or os.path.join("out", f"文本谱_{a.song}.txt")
    open(out, "w", encoding="utf8").write(s)
    print("写出:", out)
    return out


if __name__ == "__main__":
    main()
