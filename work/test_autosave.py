"""验证自动保存：改一个音 → 等 3 秒 → 切页 → 再打开，改动还在吗？"""
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


def wait_job():
    while True:
        j = getj("/api/job")
        if not j["running"]:
            return j
        time.sleep(0.4)


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
other = [x for x in lib["items"] if x.get("title") == "越冰船"][0]

post("/api/open", {"id": it["id"]})
wait_job()
st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
n0 = notes[0]
print("改之前 第 1 个音: digit=%s oct=%s" % (n0["digit"], n0["oct"]))
newd = "7" if n0["digit"] != "7" else "2"
post("/edit", {"id": n0["id"], "digit": newd, "oct": 0})
print("改成", newd, "，等 3 秒让自动保存落盘…")
time.sleep(3.0)

log = open(r"D:\简谱转调项目\out\roi_server.log", encoding="utf8").read().splitlines()[-6:]
print("服务器日志尾部:")
for ln in log:
    print("   ", ln)

# 切到另一页（触发切页落盘），再切回来
post("/api/open", {"id": other["id"]})
wait_job()
post("/api/open", {"id": it["id"]})
wait_job()
st2 = getj("/state")
n0b = [n for n in st2["notes"] if not n.get("u")][0]
print("重新打开后 第 1 个音: digit=%s oct=%s" % (n0b["digit"], n0b["oct"]))
print("自动保存生效:", n0b["digit"] == newd)
print("state.autosaved =", st2.get("autosaved"))
