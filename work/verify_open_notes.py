"""验证：修复后打开"之前没音符"的页，应该能看到成百上千个音符。"""
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

# 找「纯蓝」「唱山」「越冰船」的页各开一页
import collections
want = ["纯蓝", "唱山", "越冰船", "上山岗"]
seen = set()
for name in want:
    hits = [x for x in j["items"] if name in (x.get("title") or "") and x["id"] not in seen]
    if not hits:
        hits = [x for x in j["items"] if name in (x.get("song") or "") and x["id"] not in seen]
    if not hits:
        print(name, "：库里没有")
        continue
    it = hits[0]
    seen.add(it["id"])
    t0 = time.time()
    post("/api/open", {"id": it["id"]})
    while True:
        jj = getj("/api/job")
        if not jj["running"]:
            break
        time.sleep(0.5)
    st = getj("/state")
    notes = [n for n in st.get("notes") or [] if not n.get("u")]
    unc = [n for n in st.get("notes") or [] if n.get("u")]
    print("%-6s %-10s 音符 %3d  橙框 %2d   %.1fs   %s"
          % (name, it["id"][:8], len(notes), len(unc), time.time() - t0,
             "✅" if len(notes) > 20 else "❌ 还是没音符"))
