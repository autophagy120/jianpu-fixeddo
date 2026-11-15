"""验证两件事：① 自动保存真的落盘 ② Ctrl 原图跟随翻页（接口层面检查 /orig 随页变）。"""
import hashlib
import json
import os
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
        time.sleep(0.4)


lib = getj("/api/lib")
from collections import Counter
cnt = Counter(x.get("group") for x in lib["items"] if x.get("group"))
gid = [g for g, n in cnt.items() if n >= 2][0]
pages = sorted([x for x in lib["items"] if x.get("group") == gid],
               key=lambda x: x.get("group_pos") or 0)
print("测试份:", lib["groups"][gid]["name"], "页数", len(pages))

post("/api/open", {"id": pages[0]["id"]})
wait_job()
st = getj("/state")
n0 = [n for n in st["notes"] if not n.get("u")][0]
orig_digit = str(n0["digit"])

# ① 自动保存
cache = os.path.join(r"D:\简谱转调项目\library", pages[0]["id"], "cache.pkl")
m0 = os.path.getmtime(cache) if os.path.exists(cache) else 0
newd = "3" if orig_digit != "3" else "5"
post("/edit", {"id": n0["id"], "digit": newd, "oct": 0})
time.sleep(3.0)
m1 = os.path.getmtime(cache) if os.path.exists(cache) else 0
print("自动保存: cache 时间 %.1f → %.1f（变了 %s）" % (m0, m1, m1 > m0))
st2 = getj("/state")
print("  autosaved =", st2.get("autosaved"), " 第一音 =", [n for n in st2["notes"] if not n.get("u")][0]["digit"])

# 切页再回来，确认改动还在
post("/api/open", {"id": pages[1]["id"]})
wait_job()
post("/api/open", {"id": pages[0]["id"]})
wait_job()
st3 = getj("/state")
d3 = [n for n in st3["notes"] if not n.get("u")][0]["digit"]
print("切页回来第一音 =", d3, "（应为 %s）→ %s" % (newd, d3 == newd))

# ② 原图随页变化：两页的 /orig 必须不同
def orig_hash():
    data = urllib.request.urlopen(B + "/orig", timeout=300).read()
    return hashlib.sha1(data).hexdigest()[:12], len(data)

h1 = orig_hash()
print("第 1 页 /orig:", h1)
post("/api/open", {"id": pages[1]["id"]})
wait_job()
h2 = orig_hash()
print("第 2 页 /orig:", h2, " 不同:", h1[0] != h2[0])

# 还原刚才的改动
post("/api/open", {"id": pages[0]["id"]})
wait_job()
post("/edit", {"id": 0, "digit": orig_digit, "oct": 0})
time.sleep(2.5)
print("已还原第一音为", orig_digit)
