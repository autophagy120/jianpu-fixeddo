"""测：人工导入 = 一份谱多页（按文件名排序），OCR 照跑；再测插入。"""
import json
import os
import sys
import time
import urllib.request
import uuid

import cv2
import numpy as np

sys.path.insert(0, r"D:\简谱转调项目")
B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def multipart(fields, files, url):
    b = "----wb" + uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                    % (b, k, v)).encode("utf8"))
    for name, data in files:
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"files\"; filename=\"%s\"\r\n"
                    "Content-Type: image/png\r\n\r\n" % (b, name)).encode("utf8"))
        out.append(data)
        out.append(b"\r\n")
    out.append(("--%s--\r\n" % b).encode("utf8"))
    req = urllib.request.Request(url, data=b"".join(out),
                                 headers={"Content-Type": "multipart/form-data; boundary=" + b})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


for _ in range(60):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

src = [x for x in j["items"] if x.get("title") == "纯蓝"][0]["image"]
bgr = cv2.imdecode(np.frombuffer(open(src, "rb").read(), np.uint8), cv2.IMREAD_COLOR)
imgs = []
for k, (y0, y1) in enumerate(((0, 700), (700, 1400)), 1):
    crop = bgr[y0:y1, :]
    ok, buf = cv2.imencode(".png", crop)
    imgs.append(("page_%02d.png" % k, buf.tobytes()))

t0 = time.time()
r = multipart({"title": "两页演示谱", "key": "D", "target": "C"}, imgs, B + "/api/import_manual")
print("提交:", r, "%.1fs" % (time.time() - t0))
while True:
    jj = getj("/api/job")
    if not jj.get("running"):
        print("结束:", jj.get("error") or jj.get("msg"))
        break
    time.sleep(1.5)

lib = getj("/api/lib")
hits = [x for x in lib["items"] if x.get("title") == "两页演示谱"]
hits.sort(key=lambda x: x.get("group_pos") or 0)
print("页数:", len(hits))
for h in hits:
    gname = (lib["groups"].get(h.get("group") or "") or {}).get("name")
    print("  第%s页 id=%s key=%s status=%s ocr=%s 组=%s"
          % (h.get("group_pos"), h["id"][:6], h.get("key"), h.get("status"),
             bool(h.get("ocr_txt")), gname))

# 打开第一页测插入
post("/api/open", {"id": hits[0]["id"]})
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        break
    time.sleep(0.4)
st = getj("/state")
n0 = len([n for n in st["notes"] if not n.get("u")])
r = post("/insert", {"x": 500, "y": 800, "digit": "5", "oct": 0})
st2 = getj("/state")
n1 = len([n for n in st2["notes"] if not n.get("u")])
print("插入：音符 %d → %d  ok=%s" % (n0, n1, r.get("ok")))

# 清理演示
for h in hits:
    post("/api/delete", {"id": h["id"]})
print("演示页已移到垃圾箱")
