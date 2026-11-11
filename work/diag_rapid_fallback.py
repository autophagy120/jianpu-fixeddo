"""试：VLM 只吐歌词的页，用 RapidOCR 的数字框重建谱行，看能不能对齐出音符。"""
import os
import sys
import collections

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.ocr_boxes as OB
from jp.glyphnet import GlyphNet
import jp.merge_octave as MO


def rapid_lines(img_path):
    bgr = L.decode_bgr(img_path)
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes, _ = OB.read_boxes(gray)
    chars = OB.chars_from_boxes(boxes, digits_only=False)
    digits = [c for c in chars if str(c.ch).isdigit()]
    if not digits:
        return [], len(chars)
    hs = sorted(c.h for c in digits)
    med = hs[len(hs) // 2]
    rows = []
    for c in sorted(digits, key=lambda c: c.cy):
        for r in rows:
            if abs(r["y"] - c.cy) <= 0.6 * med:
                r["items"].append(c)
                r["y"] = sum(x.cy for x in r["items"]) / len(r["items"])
                break
        else:
            rows.append(dict(y=c.cy, items=[c]))
    rows.sort(key=lambda r: r["y"])
    lines = []
    for r in rows:
        r["items"].sort(key=lambda c: c.cx)
        lines.append(" ".join(str(c.ch) for c in r["items"]))
    return lines, len(chars)


for i in ("03387ef7", "1bb26559", "d529bb28"):
    it = [x for x in L.load().values() if x["id"].startswith(i)][0]
    lines, nchars = rapid_lines(it["image"])
    print("\n%s  RapidOCR: %d 字符, %d 行" % (i, nchars, len(lines)))
    for ln in lines[:6]:
        print("   ", ln[:70])
    txt = "\n".join(lines)
    if not lines:
        continue
    net = GlyphNet()
    try:
        _l, pairs, st, page = MO.merge_page(it["image"], txt, net)
        print("   merge: pairs=%d agree=%.2f" % (len(pairs), (st or {}).get("agree_rate", -1)))
    except Exception as e:
        print("   merge 失败:", e)
