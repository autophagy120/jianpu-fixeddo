"""重跑 VLM OCR，对比 raw / sanitize / trim 三个阶段，看谱行是在哪一步没的。"""
import os
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.merge_octave as MO
import jp.xiaomi_ocr as XO


def digit_lines(txt):
    return [ln for ln in (txt or "").splitlines() if sum(c.isdigit() for c in ln) >= 3]


for pref in ("37d477b5", "348812b9", "b3c2301b"):
    it = [x for x in L.load().values() if x["id"].startswith(pref)][0]
    print("=" * 70)
    print("页:", it["id"][:8], it.get("song"), it.get("title"), " image=", os.path.basename(it["image"]))
    raw = XO.ocr_image(it["image"])
    san = MO.sanitize_ocr(raw)
    tri, cut = MO.trim_degenerate(san)
    print("  raw   : %d 字, %d 行数字行" % (len(raw), len(digit_lines(raw))))
    print("  sanit : %d 字, %d 行数字行" % (len(san), len(digit_lines(san))))
    print("  trim  : %d 字, %d 行数字行 (截掉 %s 行)" % (len(tri), len(digit_lines(tri)), cut))
    print("  ---- raw 前 700 字 ----")
    print(raw[:700])
    print("  ---- raw 里的数字行（前 6 行） ----")
    for ln in digit_lines(raw)[:6]:
        print("   ", ln[:80])
    # 存盘文本对比
    tp = it.get("ocr_txt") or ""
    if tp and os.path.exists(tp):
        cur = open(tp, encoding="utf8").read()
        print("  存盘文本: %d 字, %d 行数字行" % (len(cur), len(digit_lines(cur))))
    print()
