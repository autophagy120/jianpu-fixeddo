"""全库批量出图，并统计质量（音符数/橙框数），挑出最需要人工看的页。"""
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


t0 = time.time()
print("启动批量出图:", post("/api/process_all", {"mode": "render"}))
last = ""
while True:
    j = getj("/api/job")
    m = j.get("msg") or ""
    if m != last:
        last = m
        print("  ", m, "(%s/%s)" % (j.get("done"), j.get("total")))
    if not j["running"]:
        print("结束:", j.get("error") or m, " 用时 %.0fs" % (time.time() - t0))
        break
    time.sleep(2)

lib = getj("/api/lib")
items = lib["items"]
done = [x for x in items if x.get("out")]
print("\n已出图 %d / %d 页" % (len(done), len(items)))
