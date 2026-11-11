"""验证修正记录：改音/回填/删除/撤销 各自留下的事件与字形样本。"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, r"D:\简谱转调项目")
B = "http://127.0.0.1:8769"
import jp.edits as ED


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

before = ED.stats()
print("改之前：事件 %d，样本 %d" % (before["events"], before["samples"]))

it = [x for x in j["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        break
    time.sleep(0.4)

st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
unc = [n for n in st["notes"] if n.get("u")]

# 1) 改 3 个音
for n in notes[:3]:
    post("/edit", {"id": n["id"], "digit": "4", "oct": 1})
# 2) 回填 1 个橙框
if unc:
    post("/assign", {"slot": 0, "digit": "3", "oct": 0})
# 3) 删 2 个音
post("/del_many", {"ids": [notes[5]["id"], notes[6]["id"]]})
# 4) 改字号
post("/size", {"ids": [notes[8]["id"]], "cls": "aux"})

after = ED.stats()
print("改之后：事件 %d（+%d），样本 %d（+%d）"
      % (after["events"], after["events"] - before["events"],
         after["samples"], after["samples"] - before["samples"]))
print("  按标签：", after["by_label"])
print("  按来源：", after["by_kind"])
print("  按操作：", after["by_event"])

# 5) 撤回两步看样本是否被过滤
post("/undo", {})
post("/undo", {})
xs, ys, meta = ED.load_dataset()
print("撤销后剩有效样本：%d（原始 %d）" % (len(xs), len(ED.load_index())))
print("目录：", ED.TRAIN, "  尺寸：",
      sum(os.path.getsize(os.path.join(ED.PATCH_DIR, f))
          for f in os.listdir(ED.PATCH_DIR)) // 1024, "KB")
