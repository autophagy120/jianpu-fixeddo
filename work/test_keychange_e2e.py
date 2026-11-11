"""端到端验证：中途转调只影响锚点之后的音，且图上出现紫色转调标记。"""
import io, json, time, urllib.request
import numpy as np, cv2

B = "http://127.0.0.1:8769"


def get(p):
    return urllib.request.urlopen(B + p, timeout=120).read()


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())


def img():
    return cv2.imdecode(np.frombuffer(get("/img"), np.uint8), cv2.IMREAD_COLOR)


def deepeq(a, b):
    return a.shape == b.shape and int(np.abs(a.astype(int) - b.astype(int)).sum())


# --- 等服务起来
for i in range(90):
    try:
        st = json.loads(get("/state"))
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("服务没起来")

print("原调", st["source_key"], "目标调", st["target_key"], "转调点", len(st["key_changes"]))
notes = [n for n in st["notes"] if not n.get("u")]
print("已识别音符", len(notes))

# 选一行中间的音做锚点
rows = {}
for n in notes:
    rows.setdefault(n["row"], []).append(n)
row = sorted(rows)[len(rows) // 3]           # 取大约 1/3 处的行
ns = sorted(rows[row], key=lambda n: n["x"])
mid = ns[len(ns) // 2]
ax = float(mid["x"] + mid["w"] / 2.0)
ay = float(mid["y"] + mid["h"] / 2.0)
print("锚点：行%d  第%d/%d 个音  (x=%.0f, y=%.0f)" % (row, len(ns) // 2, len(ns), ax, ay))
row_ys = sorted(set(round(n["y"]) for n in notes))
y_this = min(row_ys, key=lambda y: abs(y - ay))

img0 = img()
r = post("/keychange", {"x": ax, "y": ay, "key": "D"})
print("打转调点:", r)
st = json.loads(get("/state"))
print("状态里的转调点:", st["key_changes"])
img1 = img()

# 紫色标记像素（BGR 约 150,40,200）
def purple(a):
    Bc, Gc, Rc = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    return int(((Rc > 120) & (Rc < 220) & (Bc > 100) & (Gc < 100)).sum())


print("紫色标记像素: 前 %d → 后 %d" % (purple(img0), purple(img1)))

H, W = img0.shape[:2]
far = 60                                  # 躲开标记柱


def region(a, y0, y1, x0, x1):
    return a[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]


# 1) 锚点所在行**左侧**（远离标记列）：应当完全没变
left = region(img0, int(ay - 40), int(ay + 40), 0, max(0, int(ax - far)))
left1 = region(img1, int(ay - 40), int(ay + 40), 0, max(0, int(ax - far)))
print("锚点行左侧 diff =", deepeq(left, left1), "(0 = 完全一致)")

# 2) 锚点所在行**右侧**：应当变了（至少有些音换了数字）
right = region(img0, int(ay - 40), int(ay + 40), int(ax + far), W)
right1 = region(img1, int(ay - 40), int(ay + 40), int(ax + far), W)
print("锚点行右侧 diff =", deepeq(right, right1), "(>0 = 变了)")

# 3) 锚点**上方**的行：应当没变（让开标记标签，它画在标记柱上方约 3 个字高内）
above0 = region(img0, 0, max(0, int(ay - 100)), 0, W)
above1 = region(img1, 0, max(0, int(ay - 100)), 0, W)
print("上方各行 diff =", deepeq(above0, above1), "(0 = 完全一致)")

# 4) 锚点**下方**的行：应当变了
below0 = region(img0, int(ay + 60), H, 0, W)
below1 = region(img1, int(ay + 60), H, 0, W)
print("下方各行 diff =", deepeq(below0, below1), "(>0 = 变了)")

# 5) 目标调：1=G → 1=C（全谱重写）
r = post("/target", {"key": "C"})
st = json.loads(get("/state"))
img2 = img()
print("目标调改为", st["target_key"], ":", r, " 全图 diff =", deepeq(img1, img2))

# 6) 回退
post("/undo", {})
post("/undo", {})
post("/undo", {})
st = json.loads(get("/state"))
img3 = img()
print("回退后：原调", st["source_key"], "目标调", st["target_key"],
      "转调点", len(st["key_changes"]), " 与基线 diff =", deepeq(img0, img3), "(0 = 完全复原)")
