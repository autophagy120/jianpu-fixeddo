"""端到端验证主/辅助字号：设成辅助后字形高度应缩到约 0.72，恢复后回到原样。"""
import json, os, sys, time, urllib.request
import numpy as np, cv2

B = "http://127.0.0.1:8769"


def get(p):
    return urllib.request.urlopen(B + p, timeout=180).read()


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


def img():
    return cv2.imdecode(np.frombuffer(get("/img"), np.uint8), cv2.IMREAD_COLOR)


def ink_h(bgr, box, pad=6):
    """量某个音框里红字的墨迹高度（底边对齐，缩字号应把顶边压低）。"""
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = bgr[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    R, G, Bc = sub[:, :, 2].astype(int), sub[:, :, 1].astype(int), sub[:, :, 0].astype(int)
    m = (R > 150) & (G < 80) & (Bc < 80)
    ys = np.nonzero(m.any(axis=1))[0]
    return int(ys.max() - ys.min() + 1) if len(ys) else 0


for _ in range(90):
    try:
        st = json.loads(get("/state"))
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("服务没起来")

notes = [n for n in st["notes"] if not n.get("u")]
rows = {}
for n in notes:
    rows.setdefault(n["row"], []).append(n)
row = sorted(rows)[2]
ns = sorted(rows[row], key=lambda n: n["x"])[:4]
ids = [n["id"] for n in ns]
boxes = [n["box"] if "box" in n else (n["x"], n["y"], n["x"] + n["w"], n["y"] + n["h"]) for n in ns]
print("行%d 取 %d 个音：ids=%s" % (row, len(ids), ids))

a = img()
h0 = [ink_h(a, b) for b in boxes]
print("原始字高    ", h0)

print("设辅助字号:", post("/size", {"ids": ids, "cls": "aux"}))
b = img()
h1 = [ink_h(b, bb) for bb in boxes]
print("辅助字高    ", h1)
ratios = [round(h1[i] / h0[i], 2) if h0[i] else 0 for i in range(len(h0))]
print("高度比例    ", ratios, "（期望≈0.72）")

print("整行恢复主字号:", post("/size_row", {"row": row, "cls": "main"}))
c = img()
h2 = [ink_h(c, bb) for bb in boxes]
print("恢复后字高  ", h2, " 与原始一致:", h2 == h0)

st2 = json.loads(get("/state"))
aux_n = sum(1 for n in st2["notes"] if n.get("sc") == "aux")
print("状态里 aux 个数:", aux_n)

# 清场
post("/undo", {})
post("/undo", {})
st3 = json.loads(get("/state"))
print("回退后 aux 个数:", sum(1 for n in st3["notes"] if n.get("sc") == "aux"))
ok = (h2 == h0 and all(0.55 <= r <= 0.85 for r in ratios if r) and aux_n == 0)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
