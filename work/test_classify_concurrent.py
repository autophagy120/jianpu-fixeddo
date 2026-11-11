"""验证 10 并发分类 + 联网识别质量：只跑 12 页做样板。"""
import json
import os
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

# 选 12 页有 OCR 的，且属于不同歌曲
items = j["items"]
pick, songs = [], set()
for it in items:
    tp = it.get("ocr_txt") or ""
    if not tp or not os.path.exists(tp):
        continue
    if it.get("song") in songs and len(pick) >= 6:
        continue
    songs.add(it.get("song"))
    pick.append(it)
    if len(pick) >= 12:
        break
ids = [x["id"] for x in pick]
print("选中 %d 页、%d 首歌" % (len(ids), len(songs)))
post("/api/titles_clear", {"ids": ids})

t0 = time.time()
print("启动 10 并发分类:", post("/api/classify", {"ids": ids}))
last = ""
while True:
    jj = getj("/api/job")
    if jj["msg"] != last:
        last = jj["msg"]
        print("  ", last)
    if not jj["running"]:
        print("   结果:", jj["error"] or jj["msg"], " 用时 %.1fs" % (time.time() - t0))
        break
    time.sleep(1)

j2 = getj("/api/lib")
got = [x for x in j2["items"] if x["id"] in set(ids)]
print("\n结果：")
for x in got:
    print("  %-18s -> %-24s" % ((x.get("song") or "")[:18], (x.get("title") or "(未标)")[:24]))
print("有标题 %d / %d" % (sum(1 for x in got if x.get("title")), len(got)))
