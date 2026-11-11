"""量局部刷新的实际收益：改一个音 / 删一个音 / 取补丁。"""
import json
import time
import urllib.request

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
t0 = time.time()
post("/api/open", {"id": it["id"]})
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        break
    time.sleep(0.4)
print("打开: %.1fs" % (time.time() - t0))

st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
nid = notes[5]["id"]
box = [notes[5]["x"], notes[5]["y"], notes[5]["w"], notes[5]["h"]]

# 旧路径：改音 + 整页 /img
t0 = time.time()
post("/edit", {"id": nid, "digit": "3", "oct": 0})
im = urllib.request.urlopen(B + "/img", timeout=300).read()
t_full = time.time() - t0
print("改音 + 整页图: %.3fs  (%d KB)" % (t_full, len(im) // 1024))

# 新路径：改音 + 局部 patch
t0 = time.time()
post("/edit", {"id": nid, "digit": "5", "oct": 0})
X, Y, W, H = int(box[0]) - 90, int(box[1]) - 90, int(box[2]) + 180, int(box[3]) + 180
pr = urllib.request.urlopen(B + "/patch?x=%d&y=%d&w=%d&h=%d" % (X, Y, W, H), timeout=120).read()
t_patch = time.time() - t0
print("改音 + 局部补丁: %.3fs  (%d KB)" % (t_patch, len(pr) // 1024))
print("提速: %.1f×" % (t_full / max(1e-6, t_patch)))

# 连续改 10 个音（模拟真实纠错节奏）
t0 = time.time()
for k in range(10):
    n = notes[k]
    post("/edit", {"id": n["id"], "digit": "6", "oct": 0})
    urllib.request.urlopen(B + "/patch?x=%d&y=%d&w=300&h=200"
                           % (int(n["x"]) - 90, int(n["y"]) - 90), timeout=120).read()
print("连续改 10 个音: 共 %.2fs，平均 %.3fs/个" % (time.time() - t0, (time.time() - t0) / 10))
