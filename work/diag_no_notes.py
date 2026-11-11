"""诊断：为什么很多页"已OCR"但点进去没有音符（只有橙框）。"""
import os
import re
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.merge_octave as MO

d = L.load()
items = [x for x in d.values() if not x.get("deleted") and x.get("ocr_txt")
         and os.path.exists(x.get("ocr_txt") or "")]

print("有 OCR 文本的页:", len(items))

# 先做纯文本体检：有没有"像谱行的数字行"
def music_like_lines(txt):
    n = 0
    for ln in txt.splitlines():
        digits = sum(1 for c in ln if c.isdigit())
        if digits >= 3:
            n += 1
    return n

stats = []
for it in items:
    txt = open(it["ocr_txt"], encoding="utf8").read()
    nl = len([l for l in txt.splitlines() if l.strip()])
    ml = music_like_lines(txt)
    stats.append((ml, nl, it))

stats.sort(key=lambda x: x[0])
print("\n数字行最少的 10 页（大概率有问题）：")
for ml, nl, it in stats[:10]:
    print("  数字行 %2d / 非空行 %2d   %-10s %-18s %s"
          % (ml, nl, it["id"][:8], (it.get("song") or "")[:18],
             (it.get("title") or "(无标题)")[:16]))
print("\n数字行 >=3 的页数:", sum(1 for ml, _, _ in stats if ml >= 3), "/", len(stats))

# 挑一个数字行少的页真跑一遍 merge，看 pairs 有多少
from jp.glyphnet import GlyphNet
net = GlyphNet()
for ml, nl, it in stats[:3]:
    try:
        lines, pairs, st, page = MO.merge_page(it["image"], open(it["ocr_txt"], encoding="utf8").read(), net)
        print("\n%s  (数字行 %d)  → pairs=%d  stats=%s" %
              (it["id"][:8], ml, len(pairs),
               {k: st.get(k) for k in ("agree_rate", "coverage") if isinstance(st, dict)}))
        print("   OCR 前 160 字:", repr(open(it["ocr_txt"], encoding="utf8").read()[:160]))
    except Exception as e:
        print(it["id"][:8], "merge 失败:", e)
