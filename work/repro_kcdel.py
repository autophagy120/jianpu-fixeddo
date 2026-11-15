"""复现：加一个转调点再删掉，红字是否整页消失。"""
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


def red_pixels():
    d = urllib.request.urlopen(B + "/img", timeout=600).read()
    b = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)
    r = b[:, :, 2].astype(int)
    g = b[:, :, 1].astype(int)
    bl = b[:, :, 0].astype(int)
    return int(((r > 110) & (g < 95) & (bl < 95)).sum())


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
print("音符 %d  橙框 %d" % (len(notes), len(st["notes"]) - len(notes)))
print("基线红像素:", red_pixels())

# 加一个转调点在第二行中间
row2 = [n for n in notes if n["row"] == notes[0]["row"] + 1] or notes[:5]
x = row2[len(row2) // 2]["x"] + 5
y = row2[len(row2) // 2]["y"] + 5
print("加转调点 @(%d,%d)" % (x, y), post("/keychange", {"x": x, "y": y, "key": "G"}))
time.sleep(0.5)
print("  加后 state.key_changes:", getj("/state").get("key_changes"), " 红像素:", red_pixels())

ks = getj("/state").get("key_changes") or []
if ks:
    print("删除该点:", post("/keychange_del", {"i": 0}))
    time.sleep(0.5)
    st3 = getj("/state")
    n3 = len([n for n in st3["notes"] if not n.get("u")])
    print("  删后 音符:", n3, " key_changes:", st3.get("key_changes"), " 红像素:", red_pixels())
