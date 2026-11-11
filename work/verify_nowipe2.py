import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, r"D:\简谱转调项目")
B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
print("当前音符:", len(notes))
ids = [n["id"] for n in notes][:3]
post("/del_many", {"ids": ids})
after = len([x for x in getj("/state")["notes"] if not x.get("u")])
post("/undo", {})
back = len([x for x in getj("/state")["notes"] if not x.get("u")])
print("批量删除→撤销：%d → %d → %d  %s"
      % (len(notes), after, back, "✅" if back == len(notes) else "❌"))

# 存一张当前渲染图，人工看有没有整行被擦白
data = urllib.request.urlopen(B + "/img", timeout=300).read()
out = r"D:\简谱转调项目\out\_roi_nowipe.png"
open(out, "wb").write(data)
print("渲染图:", out, len(data) // 1024, "KB")

h = urllib.request.urlopen(B + "/", timeout=30).read().decode("utf8")
print("页面含 Ctrl+Z 回退:", ("ctrlKey" in h and "undo()" in h))
print("页面含 删除选中:", "delSelected" in h)
print("页面含 删除框内所有音符:", "删除框内所有音符" in h)
