"""在原谱图片上把音符替换成固定调红字（保留反复段/双行歌词等版面结构）。

用法：
    python -m jp.replace_img --orig <图片> --ocr <OCR文本> --key bA --out out/x_fixed.jpg

`--ocr` 可以是已有的 OCR 结果文本；不给就先调 Xiaomi-OCR 跑一遍
（本地、免费、约 1 分钟/页，需要 E:\\jianpu-env\\ocr 已装好）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import core
from . import emit_fixed as EF
from . import merge_octave as MO
from . import music as M
from . import xiaomi_ocr as XO


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--ocr", default="", help="已缓存的 OCR 文本；为空则现场跑")
    ap.add_argument("--key", default="", help="原调，如 bA；为空则从 OCR 头部猜")
    ap.add_argument("--target", default="", help="目标固定调 1=?；为空则自动选")
    ap.add_argument("--out", default="")
    ap.add_argument("--scale", type=float, default=1.0, help="输出放大倍数（可选）")
    args = ap.parse_args(argv)

    from .glyphnet import GlyphNet

    ocr_txt = args.ocr
    if ocr_txt and "\n" not in ocr_txt and os.path.exists(ocr_txt):
        ocr_txt = open(ocr_txt, encoding="utf8").read()   # 传的是缓存路径
    if not ocr_txt:
        if not XO.available():
            sys.exit("Xiaomi-OCR 不可用，请检查 E:\\jianpu-env\\ocr 是否装好")
        print("调用 Xiaomi-OCR …")
        ocr_txt = ""
        for attempt in range(3):
            raw = XO.ocr_image(args.orig)
            trimmed, cut = MO.trim_degenerate(raw)
            ocr_txt = trimmed if cut else raw
            if not cut:
                break
            print("  OCR 退化输出（截掉 %d 行），重试 %d/3" % (cut, attempt + 1))
        ocr_txt = MO.sanitize_ocr(ocr_txt)
        cache = os.path.splitext(args.orig)[0] + ".ocr.txt"
        open(cache, "w", encoding="utf8").write(ocr_txt)
        print("OCR 结果缓存:", cache)
    ocr_txt = MO.sanitize_ocr(ocr_txt)

    key_pc = M.parse_key(args.key) if args.key else None
    if key_pc is None:
        key_pc = KEY_FROM_OCR(ocr_txt)
        print("从谱面头部读到原调 1=%s" % M.key_name(key_pc))

    lines, pairs, stat, page = MO.merge_page(args.orig, ocr_txt, GlyphNet())
    print("对齐统计:", stat, " 配对数:", len(pairs))

    all_notes = [n for l in lines for n in l]
    pitches = [M.note_to_abs(int(n.digit), n.acc, n.octave, key_pc) for n in all_notes]
    if args.target:
        ref_pc = M.parse_key(args.target)
        reason = "指定"
    else:
        ref_pc, reason = M.pick_reference(pitches, 3)
    print("固定调 1=%s  (%s)" % (M.key_name(ref_pc), reason))

    bgr, st = EF.render_fixed_image(page, pairs, key_pc, ref_pc)
    print("绘制:", st)
    out = args.out or (os.path.splitext(args.orig)[0] + "_fixed.jpg")
    if args.scale and abs(args.scale - 1.0) > 1e-3:
        bgr = cv2_resize(bgr, args.scale)
    core.imwrite(out, bgr)
    print("写出:", out)
    return out


def KEY_FROM_OCR(txt: str) -> int:
    """从谱面头部读原调。返回 0(C) 表示没读到。

    注意 OCR 会用音乐符号写调号：`1=♭A`、`1=♯F`，不是 ASCII 的 `b`/`#`。
    只认 `[#b]` 会把 bA 调读成 C 调——整首移高五度而不自知，是很严重的错。
    """
    import re
    flat = "♭ｂb\u266d"
    sharp = "♯＃#\u266f"
    m = re.search(r"1\s*[=＝]\s*([" + flat + sharp + r"]?)\s*([A-Ga-g])", txt)
    if m:
        acc = m.group(1)
        name = m.group(2).upper()
        if acc in flat:
            name = "b" + name
        elif acc in sharp:
            name = "#" + name
        try:
            return M.parse_key(name)
        except Exception:
            pass
    return 0          # 兜底 C


def cv2_resize(bgr: np.ndarray, k: float) -> np.ndarray:
    import cv2
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (int(round(w * k)), int(round(h * k))),
                      interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)


if __name__ == "__main__":
    main()
