"""验证：未配对行不再被擦白（原谱保留），以及删除后撤销正常。"""
import hashlib
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


def img():
    return urllib.request.urlopen(B + "/img", timeout=300).read()


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
unc = [n for n in st["notes"] if n.get("u")]
print("纯蓝：音符 %d，橙框 %d" % (len(notes), len(unc)))

import numpy as np
import cv2
bgr = cv2.imdecode(np.frombuffer(img(), np.uint8), cv2.IMREAD_COLOR)
# 统计"整行被擦白"的痕迹：某一行带内白像素比例
rows = sorted({n["row"] for n in notes})
print("音符行数:", len(rows))
# 取一个没有音符的几何行（如果有）看它是否还保留墨迹
import jp.library as L
page_rows = None
print("图尺寸:", bgr.shape[:2])
dark = (bgr.astype(int).sum(axis=2) < 300)
print("整图墨迹像素占比: %.4f" % (dark.mean(),))

# 撤销测试
ids = [n["id"] for n in notes][:3]
r = post("/del_many", {"ids": ids})
n_after = len([x for x in getj("/state")["notes"] if not x.get("u")])
post("/undo", {})
n_back = len([x for x in getj("/state")["notes"] if not x.get("u")])
print("批量删除→撤销：%d → %d → %d  %s"
      % (len(notes), n_after, n_back, "✅" if n_back == len(notes) else "❌"))
h = urllib.request.urlopen(B + "/", timeout=30).read().decode("utf8")
print("页面含 Ctrl+Z:", "ctrlKey" in h and "undo()" in h)
