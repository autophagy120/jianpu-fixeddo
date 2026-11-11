"""批量：把语料里每张原始谱面 → 就地红字固定调版本 + 文本谱 + 汇总表。

resumable：已有输出的页会跳过；OCR 结果缓存下来，重跑不用再算。

用法：
    python -m jp.batch                      # 跑全部原始页
    python -m jp.batch --limit 5            # 先跑 5 页看看
    python -m jp.batch --per-song 1         # 每首只跑 1 页（快速通览）
    python -m jp.batch --songs 三月雨,凉雨   # 只跑指定歌
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import traceback
from typing import List, Optional, Tuple

from . import core
from . import emit_fixed as EF
from . import merge_octave as MO
from . import music as M
from . import replace_img as RI
from . import xiaomi_ocr as XO

CORPUS = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱"
OUTDIR = r"out\batch"
_IMG = (".jpg", ".jpeg", ".png")


def safe(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\r\n\t]', "_", name)
    return s.strip(" .")[:120] or "unnamed"


def collect(root: str, songs: Optional[List[str]] = None,
            per_song: int = 0) -> List[Tuple[str, str]]:
    """返回 [(song, image_path)]，只要原始页（排除 `(原调X-转调C)` 对照图）。"""
    out: List[Tuple[str, str]] = []
    for dirpath, _dirs, files in os.walk(root):
        song = os.path.basename(dirpath)
        if song == os.path.basename(root):
            continue
        if songs and song not in songs:
            continue
        got = sorted(f for f in files
                     if f.lower().endswith(_IMG) and "(原调" not in f)
        if per_song:
            got = got[:per_song]
        for f in got:
            out.append((song, os.path.join(dirpath, f)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=CORPUS)
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-song", type=int, default=0)
    ap.add_argument("--songs", default="")
    ap.add_argument("--key", default="", help="强制指定原调（默认从谱头读）")
    args = ap.parse_args(argv)

    songs = [s.strip() for s in args.songs.split(",") if s.strip()] if args.songs else None
    items = collect(args.root, songs, args.per_song)
    if args.limit:
        items = items[:args.limit]
    total = len(items)
    print(f"共 {total} 页", flush=True)
    if not XO.available():
        sys.exit("Xiaomi-OCR 不可用")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "_ocr"), exist_ok=True)
    sum_path = os.path.join(args.out, "summary.tsv")
    new_file = not os.path.exists(sum_path)
    fh = open(sum_path, "a", encoding="utf8", newline="")
    wr = csv.writer(fh, delimiter="\t")
    if new_file:
        wr.writerow(["song", "page", "key", "notes", "rest", "uncovered",
                     "agree_pct", "seconds", "status", "out"])

    from .glyphnet import GlyphNet
    net = GlyphNet()
    t_all = time.time()
    ok = skip = fail = 0
    for i, (song, path) in enumerate(items, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        dst_dir = os.path.join(args.out, safe(song))
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, safe(name) + "_fixed.jpg")
        stem = safe(song + "__" + name)
        ocr_cache = os.path.join(args.out, "_ocr", stem + ".txt")
        tag = f"[{i}/{total}] {song} / {name}"
        if os.path.exists(dst):
            skip += 1
            print(tag, "→ 已有输出，跳过", flush=True)
            continue
        t0 = time.time()
        try:
            if os.path.exists(ocr_cache):
                ocr_txt = open(ocr_cache, encoding="utf8").read()
            else:
                ocr_txt = ""
                for attempt in range(3):
                    raw = MO.sanitize_ocr(XO.ocr_image(path))
                    trimmed, cut = MO.trim_degenerate(raw)
                    ocr_txt = trimmed if cut else raw
                    if not cut:
                        break
                    print(tag, f"→ OCR 退化（截掉 {cut} 行），重试 {attempt+1}/3", flush=True)
                open(ocr_cache, "w", encoding="utf8").write(ocr_txt)
            key_pc = M.parse_key(args.key) if args.key else RI.KEY_FROM_OCR(ocr_txt)
            if not args.key and key_pc == 0 and "1" not in ocr_txt[:200]:
                print(tag, "→ 警告：谱头没读到调号，按 C 处理", flush=True)
            lines, pairs, stat, page = MO.merge_page(path, ocr_txt, net)
            notes = [n for l in lines for n in l]
            pitches = [M.note_to_abs(int(n.digit), n.acc, int(n.octave), key_pc)
                       for n in notes if n.digit != "0"]
            ref_pc = M.parse_key(args.key) if args.key else None
            if ref_pc is None:
                ref_pc, _reason = M.pick_reference(pitches, 3) if pitches else (0, "")
            bgr, st = EF.render_fixed_image(page, pairs, key_pc, ref_pc)
            core.imwrite(dst, bgr)
            dt = time.time() - t0
            wr.writerow([song, name, M.key_name(key_pc), st.get("total", 0),
                         st.get("rest", 0), st.get("uncovered", 0),
                         f"{stat['agree_rate']*100:.1f}", f"{dt:.0f}", "ok",
                         os.path.relpath(dst, args.out)])
            fh.flush()
            ok += 1
            print(tag, f"→ 1={M.key_name(key_pc)} 音{st.get('total',0)} "
                       f"休止{st.get('rest',0)} 漏{st.get('uncovered',0)} "
                       f"一致{stat['agree_rate']*100:.0f}% {dt:.0f}s", flush=True)
        except Exception as e:
            fail += 1
            wr.writerow([song, name, "", "", "", "", "", f"{time.time()-t0:.0f}",
                         "ERR:" + str(e)[:120], ""])
            fh.flush()
            print(tag, "→ 失败:", e, flush=True)
            traceback.print_exc()
    fh.close()
    print(f"\n完成 ok={ok} skip={skip} fail={fail}  用时 {(time.time()-t_all)/60:.1f} 分钟",
          flush=True)
    print("汇总:", sum_path, flush=True)


if __name__ == "__main__":
    main()
