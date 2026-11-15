"""验证：把"已有一个低音点"的音改成两个低音点 → 原来那个黑点应被擦掉、换成红点。"""
import json
import time
import urllib.request

import cv2
import numpy as np

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


def img():
    d = urllib.request.urlopen(B + "/img", timeout=600).read()
    return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)


def region(b, box, pad=22):
    x0, y0, x1, y1 = [int(v) for v in box]
    return b[max(0, y0 - pad):y1 + pad, max(0, x0 - 10):x1 + 10]


def counts(sub):
    r = sub[:, :, 2].astype(int)
    g = sub[:, :, 1].astype(int)
    bl = sub[:, :, 0].astype(int)
    red = ((r > 110) & (g < 95) & (bl < 95)).sum()
    dark = ((r < 100) & (g < 100) & (bl < 100)).sum()
    return int(red), int(dark)


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
notes = [n for n in getj("/state")["notes"] if not n.get("u")]
cand = next((n for n in notes if n["oct"] == -1 and n["digit"] != "0"), None)
if cand is None:
    cand = next((n for n in notes if n["oct"] == 1 and n["digit"] != "0"), None)
if cand is None:
    print("这页没有带八度点的音")
    raise SystemExit(0)

box = [cand["x"], cand["y"], cand["x"] + cand["w"], cand["y"] + cand["h"]]
print("目标: #%s digit=%s oct=%s" % (cand["id"], cand["digit"], cand["oct"]))
r0, d0 = counts(region(img(), box))
print("改前：红 %d 黑 %d" % (r0, d0))

new_oct = cand["oct"] - 1
post("/oct", {"id": cand["id"], "oct": new_oct})
time.sleep(0.9)
r1, d1 = counts(region(img(), box))
print("改成 oct=%d 后：红 %d 黑 %d" % (new_oct, r1, d1))
print("红点变多:", r1 > r0, " 黑墨变少:", d1 < d0)

post("/oct", {"id": cand["id"], "oct": cand["oct"]})
time.sleep(0.6)
r2, d2 = counts(region(img(), box))
print("还原后：红 %d 黑 %d" % (r2, d2))
