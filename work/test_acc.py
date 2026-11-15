"""验证升降号：改 acc → state 里有 → 画出来的 token 带 #/b。"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, r"D:\简谱转调项目")

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
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
n0 = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("初始: digit=%s acc=%r oct=%s" % (n0["digit"], n0.get("acc"), n0["oct"]))

post("/edit", {"id": n0["id"], "digit": n0["digit"], "oct": n0["oct"], "acc": "#"})
time.sleep(0.4)
n1 = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("设 ♯ 后: digit=%s acc=%r" % (n1["digit"], n1.get("acc")))

# 改八度时不能丢掉升降号
post("/oct", {"id": n0["id"], "oct": -1})
time.sleep(0.4)
n2 = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("再点低1: digit=%s acc=%r oct=%s" % (n2["digit"], n2.get("acc"), n2["oct"]))

post("/edit", {"id": n0["id"], "digit": n0["digit"], "oct": n2["oct"], "acc": "b"})
time.sleep(0.4)
n3 = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("改成 ♭: acc=%r" % (n3.get("acc"),))

# 还原
post("/edit", {"id": n0["id"], "digit": n0["digit"], "oct": n0["oct"], "acc": n0.get("acc") or ""})
time.sleep(0.4)
n4 = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("已还原: digit=%s acc=%r oct=%s" % (n4["digit"], n4.get("acc"), n4["oct"]))

# 插入带升降号
r = post("/insert", {"x": 600, "y": 800, "digit": "3", "oct": 0, "acc": "#"})
new = [n for n in getj("/state")["notes"] if not n.get("u") and n.get("acc") == "#"]
print("插入 ♯3:", r, " 找到带 # 的音:", len(new))
if new:
    post("/del", {"id": new[-1]["id"]})
    print("已删除测试插入的音")
