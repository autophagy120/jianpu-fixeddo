"""验证：翻页信息、原调整份修改、目标调仍独立。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=600).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def wait_job():
    while True:
        j = getj("/api/job")
        if not j["running"]:
            return j
        time.sleep(0.5)


lib = getj("/api/lib")
# 找一个多页的份
from collections import Counter
cnt = Counter(x.get("group") for x in lib["items"] if x.get("group"))
multi = [g for g, n in cnt.items() if n >= 3]
gid = multi[0]
pages = sorted([x for x in lib["items"] if x.get("group") == gid],
               key=lambda x: x.get("group_pos") or 0)
print("测试份:", lib["groups"][gid]["name"], "页数", len(pages))

post("/api/open", {"id": pages[1]["id"]})
print("打开第 2 页:", wait_job().get("error") or "ok")
st = getj("/state")
pg = st.get("page") or {}
print("页面信息: idx=%s total=%s prev=%s next=%s group=%s key=%s"
      % (pg.get("idx"), pg.get("total"), (pg.get("prev") or "")[:8],
         (pg.get("next") or "")[:8], pg.get("group_name"), pg.get("key")))
print("上一页是第 1 页:", pg.get("prev") == pages[0]["id"])
print("下一页是第 3 页:", pg.get("next") == pages[2]["id"])

# 改整份原调
print("改原调(整份) → 1=E:", post("/api/key", {"key": "E", "scope": "group"}))
st2 = getj("/state")
print("当前 state source_key:", st2.get("source_key"), " target:", st2.get("target_key"))
lib2 = getj("/api/lib")
g2 = lib2["groups"][gid]
keys = sorted({x.get("key") for x in lib2["items"] if x.get("group") == gid})
print("组上记的调性:", g2.get("key"), " 组内各页 key:", keys)

# 单页改回
print("改原调(仅本页) → 1=G:", post("/api/key", {"key": "G", "scope": "page"}))
lib3 = getj("/api/lib")
g3 = lib3["groups"][gid]
keys3 = sorted({x.get("key") for x in lib3["items"] if x.get("group") == gid})
print("组调性仍应为 E:", g3.get("key"), " 组内 keys:", keys3)
