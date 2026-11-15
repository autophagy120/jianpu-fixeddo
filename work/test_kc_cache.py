"""验证两件事：① 转调点存进缓存后重开是否一致 ② 清空后能从 .bak 救回来。"""
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


def red():
    d = urllib.request.urlopen(B + "/img", timeout=600).read()
    b = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)
    r, g, bl = b[:, :, 2].astype(int), b[:, :, 1].astype(int), b[:, :, 0].astype(int)
    return int(((r > 110) & (g < 95) & (bl < 95)).sum())


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
pid = it["id"]

post("/api/open", {"id": pid})
wait_job()
notes = [n for n in getj("/state")["notes"] if not n.get("u")]
base_red = red()
print("基线：音符 %d 红像素 %d" % (len(notes), base_red))

# ① 加转调点 → 等自动保存 → 重开 → 是否还在、渲染是否一致
x = notes[len(notes) // 2]["x"] + 6
y = notes[len(notes) // 2]["y"] + 6
post("/keychange", {"x": x, "y": y, "key": "G"})
time.sleep(2.5)          # 等自动保存
kc_red = red()
post("/api/open", {"id": pid})
wait_job()
st = getj("/state")
print("重开后 key_changes:", st.get("key_changes"), " 红像素:", red(), "（加后是 %d）" % kc_red)
print("  转调点保住了:", bool(st.get("key_changes")), " 渲染一致:", red() == kc_red)

# ② 清空本页 → 缓存备份 → 恢复
post("/clear_all", {})
time.sleep(2.5)
print("清空后 音符:", len([n for n in getj("/state")["notes"] if not n.get("u")]))
print("restore_bak:", post("/restore_bak", {}))
post("/api/open", {"id": pid})
wait_job()
st2 = getj("/state")
print("恢复后 音符:", len([n for n in st2["notes"] if not n.get("u")]),
      " key_changes:", st2.get("key_changes"), " 红像素:", red())

# 清理：删掉测试用的转调点
ks = st2.get("key_changes") or []
for i in range(len(ks)):
    post("/keychange_del", {"i": 0})
time.sleep(0.6)
print("已清掉测试转调点:", getj("/state").get("key_changes"))
