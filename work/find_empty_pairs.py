"""体检：哪些页的 cache.pkl 里 pairs 是空的（识别结果被清掉了）。"""
import os
import pickle
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

bad = []
for it in L.load().values():
    if it.get("deleted"):
        continue
    p = os.path.join(r"D:\简谱转调项目\library", it["id"], "cache.pkl")
    if not os.path.exists(p):
        continue
    try:
        c = pickle.load(open(p, "rb"))
    except Exception:
        continue
    n = len(c.get("pairs") or [])
    ocr = it.get("ocr_txt") or ""
    has_txt = ocr and os.path.exists(ocr) and os.path.getsize(ocr) > 100
    if n == 0 and has_txt:
        bad.append((it["id"], it.get("title") or it.get("song") or "", it.get("key"), it.get("strength")))

print("cache 里 pairs=0 但 OCR 有文本的页:", len(bad))
for b in bad:
    print("   %s %-14s key=%-4s strength=%s" % (b[0][:8], b[1][:14], b[2], b[3]))
