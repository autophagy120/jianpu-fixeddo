"""两栏工作区后端联调：自动分份 → 组管理 → 拖动移动 → 缩略图。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=180).read())


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

print("初始：%d 页，%d 组" % (len(j["items"]), len(j.get("groups") or {})))

# 1) 自动分份（如果还没有组）
if not (j.get("groups") or {}):
    post("/api/group_auto", {})
    while True:
        jj = getj("/api/job")
        if not jj["running"]:
            print("自动分份:", jj["error"] or jj["msg"])
            break
        time.sleep(0.4)

j = getj("/api/lib")
gs = j["groups"]
print("组数:", len(gs))
free = [x for x in j["items"] if not x.get("group")]
print("未分类页:", len(free))

# 2) 缩略图
it = j["items"][0]
r = urllib.request.urlopen(B + "/api/thumb?id=" + it["id"], timeout=60)
data = r.read()
print("缩略图: %s bytes=%d 大小合理=%s" % (r.headers.get("Content-Type"), len(data),
                                          1000 < len(data) < 400000))

# 3) 新建谱 + 移动 + 折叠 + 改名 + 删除
g = post("/api/group_new", {"name": "手动测试谱"})["group"]
print("新建组:", g["id"], g["name"])
if free:
    mv = post("/api/move", {"ids": [free[0]["id"]], "group": g["id"]})
    print("移入:", mv)
    jj = getj("/api/lib")
    moved = [x for x in jj["items"] if x["id"] == free[0]["id"]][0]
    print("  该页 group =", moved["group"], " group_n =", moved.get("group_n"))
print("折叠:", post("/api/group_toggle", {"id": g["id"], "collapsed": True})["group"]["collapsed"])
print("改名:", post("/api/group_rename", {"id": g["id"], "name": "改名后"})["group"]["name"])
print("移回未分类:", post("/api/move", {"ids": [free[0]["id"]] if free else [], "group": ""}))
print("删除组:", post("/api/group_del", {"id": g["id"]}), "(页应退回未分类)")
jj = getj("/api/lib")
if free:
    back = [x for x in jj["items"] if x["id"] == free[0]["id"]][0]
    print("  退回后 group =", repr(back["group"]))

# 4) 页面 HTML 里该有的元素
h = urllib.request.urlopen(B + "/lib", timeout=30).read().decode("utf8")
for k in ("已分类", "未分类", "pvShow", "grpDrop", "group_new"):
    print("  页面含 %-8s %s" % (k, k in h))
