"""验证：把一个音标成低音后，它旁边原来的黑点被擦掉、换成红点。"""
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


def stats(b, box, pad=18):
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = b[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    r = sub[:, :, 2].astype(int)
    g = sub[:, :, 1].astype(int)
    bl = sub[:, :, 0].astype(int)
    red = ((r > 110) & (g < 95) & (bl < 95)).sum()
    dark = ((r < 90) & (g < 90) & (bl < 90)).sum()
    return int(red), int(dark)


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
notes = [n for n in getj("/state")["notes"] if not n.get("u")]

# 找一个 oct=0 的音，且它下方有黑色点状墨块
b0 = img()
cand = None
for n in notes[:80]:
    x0, y0, x1, y1 = n["x"], n["y"], n["x"] + n["w"], n["y"] + n["h"]
    pad = 16
    sub = b0[int(y1):int(y1) + pad, int(x0) - 6:int(x1) + 6]
    if sub.size == 0:
        continue
    dark = (sub.sum(axis=2) < 260).sum()
    if dark >= 12:
        cand = n
        break
if not cand:
    print("没找到下方有点的音")
    raise SystemExit(0)

print("目标音: #%s digit=%s oct=%s box=(%.0f,%.0f,%.0f,%.0f)"
      % (cand["id"], cand["digit"], cand["oct"], cand["x"], cand["y"],
         cand["x"] + cand["w"], cand["y"] + cand["h"]))
box = [cand["x"], cand["y"], cand["x"] + cand["w"], cand["y"] + cand["h"]]
r0, d0 = stats(img(), box)
print("改前该音周围：红 %d  黑 %d" % (r0, d0))

post("/oct", {"id": cand["id"], "oct": -1})
time.sleep(0.8)
r1, d1 = stats(img(), box)
print("标成低音后：红 %d  黑 %d" % (r1, d1))
print("红点增加:", r1 > r0, "  黑墨减少:", d1 < d0)

# 还原
post("/oct", {"id": cand["id"], "oct": 0})
time.sleep(0.6)
print("已还原")
