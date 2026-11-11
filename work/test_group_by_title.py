"""验证新版分份：之前的两个巨型组（越冰船/光与影的对白）应被 AI 标题拆开。"""
import collections
import json
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=180).read())


for _ in range(60):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        import time
        time.sleep(2)

items = j["items"]
gs = j["groups"]
print("库: %d 页 / %d 组" % (len(items), len(gs)))
cnt = collections.Counter(x.get("group_n") or 0 for x in items if x.get("group"))
big = [(gs[g]["name"], sum(1 for x in items if x.get("group") == g))
       for g in gs]
big.sort(key=lambda t: -t[1])
print("最大的 8 份:")
for n, c in big[:8]:
    print("   %-26s %d 页" % ((n or "")[:26], c))

mine = [x for x in items if x.get("song") == "保存简谱"]
print("\n保存简谱 的 53 页现在分到 %d 份：" % len({x.get("group") for x in mine}))
c2 = collections.Counter((x.get("group_title") or "(无)") for x in mine)
for n, k in c2.most_common(12):
    print("   %-26s %d 页" % (n[:26], k))
