"""核对：纯蓝、唱山这两首到底能不能出音符。"""
import os
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.merge_octave as MO
from jp.glyphnet import GlyphNet

targets = ["纯蓝", "唱山", "上山岗", "越冰船"]
items = [x for x in L.load().values() if not x.get("deleted")]
picked = []
for name in targets:
    hits = [x for x in items if name in (x.get("title") or "") or name in (x.get("song") or "")]
    print("== %s: 找到 %d 页" % (name, len(hits)))
    for h in hits[:3]:
        picked.append((name, h))

net = GlyphNet()
for name, it in picked:
    tp = it.get("ocr_txt") or ""
    txt = open(tp, encoding="utf8").read() if tp and os.path.exists(tp) else ""
    nlines = L.music_line_count(txt)
    print("\n--- %s  id=%s  song=%s" % (name, it["id"][:8], it.get("song")))
    print("    OCR 文本 %d 字 / %d 行谱行" % (len(txt), nlines))
    print("    前 3 行:", [l[:34] for l in txt.splitlines() if l.strip()][:3])
    try:
        _l, pairs, st, page = MO.merge_page(it["image"], txt, net)
        print("    merge_page → pairs=%d  agree=%.2f  coverage=%.2f"
              % (len(pairs), (st or {}).get("agree_rate", -1), (st or {}).get("coverage", -1) or -1))
    except Exception as e:
        print("    merge 失败:", str(e)[:140])
