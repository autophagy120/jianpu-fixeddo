"""验证：坐标/谱面上下文是否被记录，以及 move_note 的坐标修正样本。"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, r"D:\简谱转调项目")
import jp.edits as ED

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


for _ in range(60):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

it = [x for x in j["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        break
    time.sleep(0.4)

st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
n = notes[10]
print("移动前：音符 #%d 在 (%.0f, %.0f)" % (n["id"], n["x"], n["y"]))
r = post("/move_note", {"id": n["id"], "dx": 7, "dy": -4})
print("move_note:", {k: r[k] for k in r if k != "bbox"}, " bbox:", r.get("bbox"))
st2 = getj("/state")
n2 = [x for x in st2["notes"] if x["id"] == n["id"]][0]
print("移动后：音符 #%d 在 (%.0f, %.0f)" % (n2["id"], n2["x"], n2["y"]))

# 再看样本里的 ctx
idx = ED.load_index()
last = idx[-1]
print("\n最新样本：")
for k in ("kind", "label", "pred", "box", "box_old", "dx", "dy"):
    if k in last:
        print("  %-8s %s" % (k, last[k]))
ctx = last.get("ctx") or {}
print("  ctx 键：", sorted(ctx.keys()))
print("  ctx 示例：", {k: ctx[k] for k in ("w", "h", "scale", "med_h", "row", "row_base",
                                          "dx_left", "dx_right", "row_n", "key", "strength")
                      if k in ctx})
print("\n事件总数 %d，样本 %d" % (len(ED.load_events()), len(idx)))
