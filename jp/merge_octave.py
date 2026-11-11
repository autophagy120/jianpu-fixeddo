"""数字用 Xiaomi-OCR，八度点用几何检测：**全页单调对齐**后合并。

为什么这么分工：
- Xiaomi-OCR 对数字序列几乎全对，但它是文本 OCR，**点（高低八度）不是文本**，
  只在"1 带点"时恰好写成 `i`；实测 `6 1̇ 2̇ 3̇` 只读出了第一个点（写成 `6i23`），
  2、3 上面的点全丢；
- 几何检测正好相反：数字常认错，但"数字上方/下方有没有小点"很准
  （点只有 2-3 px，按位置和墨迹判定）。

对齐见 `jp/align.py`：两边各拉成一条按阅读顺序的序列做单调对齐，
不再假设 OCR 行与视觉行一一对应。

输出两份表示：
- 首调（原谱写法）：`6 1̇ 2̇ 3̇` —— 用来核对八度点识别的对不对；
- 固定调（1=C / C#）：实际转谱结果，`( )` 低八度、`【 】` 高八度。
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np

from . import align as AL
from . import core
from . import music as M
from . import xiaomi_ocr as XO


DEFAULT_ORIG = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
DEFAULT_OCR = r"out\三月雨_ocr.txt"
DEFAULT_OUT = r"out\三月雨_textscore_merged.txt"


def _norm_oct(v) -> int:
    try:
        return max(-2, min(2, int(round(float(v or 0)))))
    except Exception:
        return 0


_PC_SHARP = ["C", "#C", "D", "#D", "E", "F", "#F", "G", "#G", "A", "#A", "B"]


def _pitch_name(p) -> str:
    """绝对音名，如 F5 / bB4。用升号写法（和 1=C# 的记谱一致）。"""
    return f"{_PC_SHARP[int(p.pc) % 12]}{int(p.octave)}"


def sanitize_ocr(txt: str) -> str:
    """去掉模型尾部会吐出来的 prompt 回显（`<|im_start|>system ...`）。"""
    for stop in ("<|im_start|>", "<|im_end|>", "[end of text]"):
        k = txt.find(stop)
        if k >= 0:
            txt = txt[:k]
    return txt


def trim_degenerate(txt: str, max_rep: int = 6) -> Tuple[str, int]:
    """截掉模型的退化循环（同一行重复几十次）。

    实测 Xiaomi-OCR-0 在同一张图上**不保证稳定**：三月雨第一次输出正常，
    第二次吐了 100 多行 `啊！`，把后面的谱行全吞了，音符数从 423 掉到 187。
    这种退化尾巴没有任何信息量，直接截掉；返回值第二项是被截掉的行数，
    非 0 就说明这一页需要重跑 OCR。
    """
    lines = txt.splitlines()
    counts: Dict[str, int] = {}
    for l in lines:
        s = l.strip()
        if s:
            counts[s] = counts.get(s, 0) + 1
    bad = {s for s, c in counts.items() if c > max_rep}
    if not bad:
        return txt, 0
    for i, l in enumerate(lines):
        if l.strip() in bad:
            return "\n".join(lines[:i]), len(lines) - i
    return txt, 0


def _looks_path(src: str) -> bool:
    """判断传进来的是"OCR 文本"还是"OCR 文件路径"。

    以前的写法是"没有换行且长度 <260 就当路径"，空文本（人工导入的页没有 OCR）
    会被当成路径去 open('')，直接 FileNotFoundError。现在必须**文件真的存在**。
    """
    if not src or "\n" in src:
        return False
    if len(src) >= 260:
        return False
    try:
        return os.path.exists(src) and os.path.isfile(src)
    except Exception:
        return False


def load_ocr_lines(src: str) -> List[List[XO.OcrNote]]:
    """`src` 可以是 OCR 文本本身，也可以是缓存文件路径。

    **休止符 `0` 保留**：它和音符一样占一个印刷位置，保留下来才能和几何槽位
    一一配对、才能也画成红字。早先把它过滤掉，结果原谱里的 `0` 一直是黑色，
    混在歌词里看不出来，和"没识别"分不清。
    """
    txt = src
    if _looks_path(src):
        txt = open(src, encoding="utf8").read()
    txt = sanitize_ocr(txt)
    parsed = [XO.parse_line(t) for t in txt.splitlines()]
    return [[XO.OcrNote(digit=n.digit, octave=_norm_oct(n.octave), acc=n.acc,
                        aug_dot=n.aug_dot) for n in l.notes]
            for l in parsed if l.is_music]


def merge_page(orig: str, ocr_txt: str, net, strength: int = 1):
    """全页对齐，返回 (每行 OCR 音符, 统计)。"""
    from .glyphnet import GlyphNet  # noqa: F401  (type hint only)

    page = core.analyse(core.load_gray(orig), net, path=orig)
    flat_slots = AL.ordered_slots(page, net)
    ocr_lines = load_ocr_lines(ocr_txt)
    txt_clean = sanitize_ocr(open(ocr_txt, encoding="utf8").read() if _looks_path(ocr_txt)
                             else ocr_txt)
    raw_lines = [l for l in txt_clean.splitlines() if XO.parse_line(l).is_music]

    # ---- 两种对齐策略，自动择优
    # A 音符级：整页拉成一条序列做单调对齐（低分辨率页的几何太弱时更稳）
    # B 小节级：小节线是印刷出来的强锚点，一处数量不符只污染一个小节；
    #           实测高清页 91% 的小节音符数完全相等，一致率 98.2%。
    #           但低分辨率页会把「数字+点粘连」的高窄块误判成小节线，反而更差。
    flat_ocr: List[XO.OcrNote] = [n for l in ocr_lines for n in l]
    line_of: List[int] = [i for i, l in enumerate(ocr_lines) for _ in l]
    mapping = AL.align_streams([n.digit for n in flat_ocr],
                               [s.digit for s in flat_slots])
    geo_by_ocr: List[Optional[core.Note]] = [
        (flat_slots[j] if j is not None else None) for j in mapping]
    strat = "note"
    try:
        from . import measures as ME
        pB, stB, ocr_m, src_m = ME.pair_by_measures(page, flat_slots, raw_lines)
        sA = ME.score_pairs([(g, n) for g, n in zip(geo_by_ocr, flat_ocr) if g is not None],
                            len(flat_ocr))
        sB = ME.score_pairs(pB, len(flat_ocr))
        if sB > sA:
            flat_ocr = [n for m in ocr_m for n in m]
            line_of = [li for m, li in zip(ocr_m, src_m) for _ in m]
            gmap = {id(n): g for g, n in pB}
            geo_by_ocr = [gmap.get(id(n)) for n in flat_ocr]
            strat = "measure"
    except Exception:
        pass

    # 去重：Xiaomi-OCR 会把带点音（如 `1̇`）写成两个相同的数字（`1 1`），
    # 实测「为了你唱下去」一行因此多出 2-3 个音，行内数量一乱，对齐就会挪位。
    # 紧挨着、数字相同、且后者在几何上没有槽位 —— 判为重复音，丢掉后者。
    dup = 0
    drop = set()
    for i in range(1, len(flat_ocr)):
        if (i not in drop and geo_by_ocr[i] is None and geo_by_ocr[i - 1] is not None
                and flat_ocr[i].digit == flat_ocr[i - 1].digit
                and flat_ocr[i].octave == 0):
            drop.add(i)
            dup += 1
    if dup:
        keep = [i for i in range(len(flat_ocr)) if i not in drop]
        flat_ocr = [flat_ocr[i] for i in keep]
        geo_by_ocr = [geo_by_ocr[i] for i in keep]
        line_of = [line_of[i] for i in keep]
    dup_removed = dup

    changed = AL.apply_octaves(flat_ocr, geo_by_ocr)

    # 切行：**以 OCR 文本行为单位**（实测 15 条 OCR 行 = 15 条视觉行，本来是对的）。
    # 只在几何证据表明该行跨了视觉行时才切：配对槽位的 y 出现 > 1.2 字高的跳变。
    # 早先用几何行带/槽位 y 聚类切行，反而把正确的 1:1 关系切碎（15 行切成 18 行，
    # 还多出 1-3 个音的碎片行），因为行带会把页眉、谱行、歌词粘成一条。
    scale = float(page.scale)
    lines: List[List[XO.OcrNote]] = []
    pos = 0
    while pos < len(flat_ocr):
        lid = line_of[pos]
        end = pos
        while end < len(flat_ocr) and line_of[end] == lid:
            end += 1
        # 这一行每个音对应槽位的 y（没配对的沿用上一个）
        ys: List[float] = []
        ly = 0.0
        for j in range(pos, end):
            g = geo_by_ocr[j]
            if g is not None:
                ly = float(g.box[1])
            ys.append(ly)
        parts: List[List[XO.OcrNote]] = [[]]
        for j in range(pos, end):
            if parts[-1] and abs(ys[j - pos] - ys[j - pos - 1]) > 1.2 * scale:
                parts.append([])
            parts[-1].append(flat_ocr[j])
        # 太短的碎片并回相邻段（真换行不会只留 1-3 个音）
        merged: List[List[XO.OcrNote]] = []
        for p in parts:
            if merged and (len(p) < 4 or len(merged[-1]) < 4):
                merged[-1].extend(p)
            else:
                merged.append(p)
        lines.extend(merged)
        pos = end

    # 量化对齐质量：对上的比例、且两边都认字时一致的比例
    paired = [i for i, g in enumerate(geo_by_ocr) if g is not None]
    agree = [i for i in paired if flat_ocr[i].digit == geo_by_ocr[i].digit]
    stat = dict(
        ocr_notes=len(flat_ocr), geo_slots=len(flat_slots),
        paired=len(paired),
        agree=len(agree),
        agree_rate=(len(agree) / len(paired)) if paired else 0.0,
        octave_changed=changed,
        octave_hits=sum(1 for n in flat_ocr if n.octave != 0),
    )
    pairs = [(g, flat_ocr[i]) for i, g in enumerate(geo_by_ocr) if g is not None]

    # ---- 补漏：几何看到、但 OCR 文本里没有对应数字的槽位，用 RapidOCR 的带框结果填。
    # 为什么需要：小字号的和声/装饰行（`(3) 3 · 4 4 1 |`）几何能切出来，
    # 但 VLM 的文本里常常把它们并进相邻行或整行省略，于是这些音只能标橙框、
    # 一个红字都不出。RapidOCR 的识别确实弱（三月雨 68.9% vs VLM 80.5%），
    # 但对这些"本来 0% 覆盖"的音，有位有字远好过空着。
    filled = 0
    # 识别强度：0=严格（只信 VLM 文本，橙框多）；1=标准（用 RapidOCR 框补漏）；
    # 2=加强（再用几何分类器给剩下的槽位补字，覆盖率最高、误识别也最多）
    if strength <= 0:
        missing = []
    try:
        covered = {id(g) for g in geo_by_ocr if g is not None}
        missing = [s for s in flat_slots if id(s) not in covered]
        if strength >= 1 and len(missing) >= 3:
            from . import ocr_boxes as OB
            stem = re.sub(r"[^\w\u4e00-\u9fff]+", "_", os.path.basename(orig))[:70]
            cache = os.path.join("out", "_ocrboxes", stem + ".json")
            chars, _secs = OB.read_page_chars(page.gray, cache)
            for slot, ch in OB.pair_positions(page, missing, chars):
                n = XO.OcrNote(digit=ch, octave=max(-2, min(2, int(slot.octave))),
                               acc="", aug_dot=False)
                pairs.append((slot, n))
                filled += 1
    except Exception:
        filled = 0

    # 标准档：RapidOCR 框没补上任何东西时，用几何分类器补（阈值更严 0.65）；
    # 加强档直接走 0.45。
    if strength >= 1 and (strength >= 2 or filled == 0):
        try:
            from . import core as _core
            have = {id(g) for g, _ in pairs}
            still = [s for s in flat_slots if id(s) not in have]
            # **只补"像音符"的槽位**：宽高比太扁/太瘦（减时线、小节线、歌词笔画）不补，
            # 否则加强档会往谱里塞一堆乱七八糟的数字。
            hs = sorted(max(1.0, s.box[3] - s.box[1]) for s in flat_slots) if flat_slots else [1.0]
            med_slot = hs[len(hs) // 2]
            cand, patches = [], []
            for s in still:
                x0, y0, x1, y1 = [int(v) for v in s.box]
                w, h = x1 - x0, y1 - y0
                if w < 3 or h < 3:
                    continue
                if h < 0.55 * med_slot or h > 1.6 * med_slot:
                    continue
                if not (0.30 <= w / float(h) <= 1.45):
                    continue
                c = _core.Comp(x0, y0, w, h, w * h)
                cand.append(s)
                patches.append(_core.patch_of(page.clean, c))
            if patches and net is not None:
                _thr = 0.45 if strength >= 2 else 0.65
                labs, confs = net.digit_with_reject(patches, min_prob=_thr)
                for s, lab, cf in zip(cand, labs, confs):
                    if not lab:
                        continue
                    pairs.append((s, XO.OcrNote(digit=lab,
                                                octave=max(-2, min(2, int(s.octave))),
                                                acc="", aug_dot=False)))
                    filled += 1
                    stat["filled_geom"] = stat.get("filled_geom", 0) + 1
        except Exception:
            pass

    stat["ocr_notes_raw"] = len(flat_ocr) + dup_removed
    stat["dup_removed"] = dup_removed
    stat["strategy"] = strat
    stat["filled_from_boxes"] = filled
    stat["strength"] = int(strength)
    try:
        page._slots = flat_slots   # 供 ROI 界面等使用：完整槽位（含未配对）
    except Exception:
        pass
    return lines, pairs, stat, page


def render_movable(notes: List[XO.OcrNote], key_pc: int) -> str:
    """首调写法，用来看八度点对不对：`6 1̇ 2̇ 3̇`。"""
    toks = []
    for n in notes:
        t = ("#" if n.acc == "#" else ("b" if n.acc == "b" else "")) + n.digit
        if n.octave > 0:
            t += "\u0307" * n.octave          # 组合上点
        elif n.octave < 0:
            t += "\u0323" * (-n.octave)       # 组合下点
        if n.aug_dot:
            t += "·"
        toks.append(t)
    return " ".join(toks)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=DEFAULT_ORIG)
    ap.add_argument("--ocr", default=DEFAULT_OCR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--key", default="bA", help="原调，如 bA / C / G")
    ap.add_argument("--lyrics", default="", help="歌词行文本，按行对齐写进来")
    args = ap.parse_args(argv)

    from .glyphnet import GlyphNet
    lines, _pairs, stat, _page = merge_page(args.orig, args.ocr, GlyphNet())
    print("对齐统计:", stat)

    key_pc = M.parse_key(args.key)
    all_notes = [n for l in lines for n in l if n.digit != "0"]
    pitches = [M.note_to_abs(int(n.digit), n.acc, n.octave, key_pc) for n in all_notes]
    ref_pc, reason = M.pick_reference(pitches, 3)
    harm = M.HarmonicaOptions()

    out = ["# 三月雨 —— 固定调文本谱",
           "# 数字: Xiaomi-OCR-0（本地）；高低八度点: 几何检测；两者全页单调对齐",
           f"# 原调 1={M.key_name(key_pc)}   固定调 1={M.key_name(ref_pc)}   {reason}",
           "# 八度：绝对音高，以 C4 为基准，( ) 低八度、【 】 高八度",
           "# 首调行是原谱写法，用来核对八度点；固定调行是成品",
           f"# 对齐: OCR {stat['ocr_notes']} 音 / 几何槽位 {stat['geo_slots']} / "
           f"配对 {stat['paired']} / 数字一致 {stat['agree_rate']*100:.1f}% / "
           f"八度补正 {stat['octave_changed']} / 带点音符 {stat['octave_hits']}",
           ""]
    for i, notes in enumerate(lines):
        if not notes:
            continue
        for n in notes:
            n.octave = _norm_oct(n.octave)
        mv = render_movable(notes, key_pc)
        # 休止符 `0` 原样保留（不转调、不参与音高统计）
        idx = [k for k, n in enumerate(notes) if n.digit != "0"]
        ps = [M.note_to_abs(int(notes[k].digit), notes[k].acc, notes[k].octave, key_pc)
              for k in idx]
        # 不开口琴优化：口琴的等音替换/八度折叠会改变书写位置，
        # 而用户要的是**绝对音高**的忠实转谱。口琴指法单独出。
        fixed_all = M.render_fixed_do(ps, ref_pc, harm, "cn", harmonica=False, octave_bias=0)
        names_all = [_pitch_name(p) for p in ps]
        fixed = ["0"] * len(notes)
        names = ["休止"] * len(notes)
        for t, nm, k in zip(fixed_all, names_all, idx):
            fixed[k] = t
            names[k] = nm
        out.append(f"第{i+1:02d}行 原谱(首调)  {mv}")
        out.append(f"        固定调1={M.key_name(ref_pc)}  " + " ".join(fixed))
        out.append("        绝对音名            " + " ".join(names))
        out.append("")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf8").write("\n".join(out))
    print("写出:", args.out)
    for line in out[5:11]:
        print(line)


if __name__ == "__main__":
    main()
