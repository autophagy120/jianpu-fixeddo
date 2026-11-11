"""查这 19 页到底是不是同一首：对比 OCR 文本文件与图片本身。"""
import hashlib
import json
import os
import urllib.request
import collections

B = "http://127.0.0.1:8769"
j = json.loads(urllib.request.urlopen(B + "/api/lib", timeout=180).read())
bad = [x for x in j["items"] if (x.get("title") or "") == "光与影的对白"]
print("页数:", len(bad))
rows = []
for x in bad:
    tp = x.get("ocr_txt") or ""
    txt = open(tp, encoding="utf8").read() if tp and os.path.exists(tp) else ""
    h = hashlib.sha1(txt.encode("utf8")).hexdigest()[:10]
    img = x.get("image") or ""
    try:
        ih = hashlib.sha1(open(img, "rb").read()).hexdigest()[:10]
        isz = os.path.getsize(img)
    except Exception:
        ih, isz = "-", 0
    rows.append((x["id"][:8], os.path.basename(tp)[:38], len(txt), h, ih, isz))
print("%-9s %-40s %6s %-11s %-11s %8s" % ("id", "ocr文件", "字数", "txt哈希", "img哈希", "图KB"))
for r in rows:
    print("%-9s %-40s %6d %-11s %-11s %8.0f" % (r[0], r[1], r[2], r[3], r[4], r[5] / 1024))
print()
print("不同 OCR 文本哈希数:", len({r[3] for r in rows}))
print("不同图片哈希数:", len({r[4] for r in rows}))
print("不同 OCR 文件数:", len({r[1] for r in rows}))
print("\n第一页 OCR 前 400 字:\n", open(bad[0]["ocr_txt"], encoding="utf8").read()[:400])
print("\n第二页 OCR 前 400 字:\n", open(bad[1]["ocr_txt"], encoding="utf8").read()[:400])
