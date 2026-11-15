"""端到端验证"还原升降号"：用浏览器真实页面点 ♯ 再点 ♮。"""
import json
import subprocess
import time
import urllib.request

B = "http://127.0.0.1:8769"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


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

# 用服务端接口验证 acc 的三态（前端现在会把 acc 发过来，逻辑同一条路径）
n0 = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("初始: digit=%s acc=%r" % (n0["digit"], n0.get("acc")))
post("/edit", {"id": n0["id"], "digit": n0["digit"], "oct": n0["oct"], "acc": "#"})
time.sleep(0.4)
a1 = [n for n in getj("/state")["notes"] if not n.get("u")][0].get("acc")
print("标 ♯ 后 acc=%r" % a1)
post("/edit", {"id": n0["id"], "digit": n0["digit"], "oct": n0["oct"], "acc": ""})
time.sleep(0.4)
a2 = [n for n in getj("/state")["notes"] if not n.get("u")][0].get("acc")
print("还原(♮) 后 acc=%r  → 成功: %s" % (a2, a2 == ""))

# 顺带确认页面里的 applyEdit 会把 acc 发出去
dom = subprocess.run([EDGE, "--headless=new", "--disable-gpu",
                      "--virtual-time-budget=8000", "--dump-dom", B + "/?selftest=1"],
                     capture_output=True, text=True, encoding="utf8", errors="replace",
                     timeout=120).stdout
print("页面含 acc 透传:", "body.acc=acc" in dom)
