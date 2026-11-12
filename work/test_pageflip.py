"""模拟翻页：open 后立刻读 state（旧行为会拿到旧页）→ 等 job 完成再读。"""
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


lib = getj("/api/lib")
from collections import Counter
cnt = Counter(x.get("group") for x in lib["items"] if x.get("group"))
gid = [g for g, n in cnt.items() if n >= 3][0]
pages = sorted([x for x in lib["items"] if x.get("group") == gid],
               key=lambda x: x.get("group_pos") or 0)
print("份:", lib["groups"][gid]["name"], " 页:", [p["id"][:8] for p in pages])

post("/api/open", {"id": pages[0]["id"]})
while getj("/api/job")["running"]:
    time.sleep(0.3)
print("已打开第 1 页:", (getj("/state").get("page") or {}).get("idx"))

# 模拟点"下一页"
t0 = time.time()
r = post("/api/open", {"id": pages[1]["id"]})
print("open 返回:", r, " 用时 %.1fs" % (time.time() - t0))
dur = time.time() - t0
print("请求返回时 job 还在跑吗:", getj("/api/job")["running"], "（旧版此时立刻 reload → 拿到旧页/占位页）")
st_now = getj("/state")
print("此刻 state 还是第", (st_now.get("page") or {}).get("idx"), "页")

# 再点一次（应当被"正忙"挡住）
r2 = post("/api/open", {"id": pages[2]["id"]})
print("连点第二次:", r2)

while getj("/api/job")["running"]:
    time.sleep(0.3)
st = getj("/state")
pg = st.get("page") or {}
print("任务完成后：第 %s/%s 页  prev=%s next=%s"
      % (pg.get("idx"), pg.get("total"), bool(pg.get("prev")), bool(pg.get("next"))))
print("确实换页了:", pg.get("idx") == 2)
