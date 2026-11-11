"""重建标题：清空全部 AI 标题 → 联网 + 10 并发重新分类 → 汇总结果。"""
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

n_title = sum(1 for x in j["items"] if x.get("title"))
print("重建前：库 %d 页，有标题 %d 页" % (len(j["items"]), n_title), flush=True)

r = post("/api/titles_clear", {"all": True})
print("清空标题：", r, flush=True)

t0 = time.time()
r = post("/api/classify", {})
print("启动联网分类：", r, flush=True)

last, last_t = "", time.time()
while True:
    jj = getj("/api/job")
    if jj["msg"] != last:
        last = jj["msg"]
        print("  [%5.0fs] %s" % (time.time() - t0, last), flush=True)
        last_t = time.time()
    if not jj["running"]:
        print("任务结束：", jj["error"] or jj["msg"], " 用时 %.1fs" % (time.time() - t0), flush=True)
        break
    if time.time() - last_t > 300:
        print("  ... 超过 5 分钟没有新进度，仍在跑", flush=True)
        last_t = time.time()
    time.sleep(2)

j2 = getj("/api/lib")
items = j2["items"]
titled = [x for x in items if x.get("title")]
print("\n重建后：库 %d 页，有标题 %d 页" % (len(items), len(titled)), flush=True)
import collections
c = collections.Counter(x["title"] for x in titled)
print("标题分布（前 15）：", flush=True)
for t, k in c.most_common(15):
    print("   %-26s %d 页" % (t[:26], k), flush=True)

print("\n与目录名不一致的例子（前 10）：", flush=True)
n = 0
for x in titled:
    s = (x.get("song") or "").strip()
    t = (x.get("title") or "").strip()
    if s and t and s not in t and t not in s:
        print("   %-22s -> %s" % (s[:22], t[:22]), flush=True)
        n += 1
        if n >= 10:
            break
