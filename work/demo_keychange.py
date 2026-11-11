"""生成一张"中途转调"演示图：1=G 原调 → 目标调 C，行 6 第 14 音起转 1=D。"""
import io, json, time, urllib.request
import numpy as np, cv2

B = "http://127.0.0.1:8769"


def get(p):
    return urllib.request.urlopen(B + p, timeout=120).read()


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())


st = json.loads(get("/state"))
notes = [n for n in st["notes"] if not n.get("u")]
rows = {}
for n in notes:
    rows.setdefault(n["row"], []).append(n)
row = sorted(rows)[len(rows) // 3]
ns = sorted(rows[row], key=lambda n: n["x"])
mid = ns[len(ns) // 2]
ax, ay = float(mid["x"] + mid["w"] / 2.0), float(mid["y"] + mid["h"] / 2.0)

post("/target", {"key": "C"})
post("/keychange", {"x": ax, "y": ay, "key": "D"})
bgr = cv2.imdecode(np.frombuffer(get("/img"), np.uint8), cv2.IMREAD_COLOR)
out = r"D:\简谱转调项目\out\BadApple_转调C_行%d起转D.png" % row
cv2.imencode(".png", bgr)[1].tofile(out)
h, w = bgr.shape[:2]
print("已保存", out, w, "x", h, " 锚点", ax, ay)
# 还原，别把演示状态留在服务里
post("/keychange_del", {"i": 0})
post("/target", {"key": st["source_key"]})
st2 = json.loads(get("/state"))
print("还原后 目标调", st2["target_key"], "转调点", len(st2["key_changes"]))
