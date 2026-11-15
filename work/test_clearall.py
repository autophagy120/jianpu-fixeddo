"""验证"全手工"链路：清空本页 → 满屏橙框 → 点橙框填音（带升降号）→ 计数正确。"""
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
        time.sleep(0.4)


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
st = getj("/state")
n0 = len([n for n in st["notes"] if not n.get("u")])
print("清空前：音符 %d，橙框 %d" % (n0, len([n for n in st["notes"] if n.get("u")])))

r = post("/clear_all", {})
time.sleep(0.8)
st2 = getj("/state")
notes2 = [n for n in st2["notes"] if not n.get("u")]
unc2 = [n for n in st2["notes"] if n.get("u")]
print("清空返回:", r, " → 音符 %d，橙框 %d（应满屏都是橙框）" % (len(notes2), len(unc2)))

if unc2:
    r2 = post("/assign", {"slot": 0, "digit": "4", "oct": 0, "acc": "#"})
    st3 = getj("/state")
    n3 = [n for n in st3["notes"] if not n.get("u")]
    print("点第一个橙框填 ♯4:", r2, " → 音符 %d，第一个是 %s%s oc=%s"
          % (len(n3), n3[0].get("acc"), n3[0]["digit"], n3[0]["oct"]))

# 回退恢复
print("撤销:", post("/undo", {}))
time.sleep(0.5)
st4 = getj("/state")
print("撤销后：音符 %d（应回到 %d）" % (len([n for n in st4["notes"] if not n.get("u")]), n0))
