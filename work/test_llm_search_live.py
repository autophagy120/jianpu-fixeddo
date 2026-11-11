"""验证联网识别 + 并发：先清空几页标题，再用 10 并发重新分类 6 页看结果与速度。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


for _ in range(60):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

print("配置:", getj("/api/llm"))
items = j["items"]
# 找几页有 OCR 文本的，不同歌曲
sample = []
seen_song = set()
import os
for it in items:
    tp = it.get("ocr_txt") or ""
    if not tp or not os.path.exists(tp):
        continue
    if it.get("song") in seen_song:
        continue
    seen_song.add(it.get("song"))
    sample.append(it)
    if len(sample) >= 6:
        break
print("取样歌曲:", [x.get("song")[:16] for x in sample])

# 清空这几页标题后单页识别，看结果和用时
for it in sample:
    post("/api/titles_clear", {"ids": [it["id"]]})

t0 = time.time()
for it in sample:
    r = post("/api/ai_meta", {"id": it["id"]})
    m = r.get("meta") or {}
    print("  %-18s -> title=%-22s conf=%-5s search=%s" %
          ((it.get("song") or "")[:18], (m.get("title") or "")[:22],
           m.get("confidence"), m.get("search")))
print("串行 6 页用时 %.1fs" % (time.time() - t0))
