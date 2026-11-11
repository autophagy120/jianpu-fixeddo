"""测：人工导入 = 只人工定标题，OCR 照跑（不调 AI）。"""
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, r"D:\简谱转调项目")
B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def multipart(fields, filename, data, url):
    b = "----wb" + uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                    % (b, k, v)).encode("utf8"))
    out.append(("--%s\r\nContent-Disposition: form-data; name=\"files\"; filename=\"%s\"\r\n"
                "Content-Type: image/png\r\n\r\n" % (b, filename)).encode("utf8"))
    out.append(data)
    out.append(("\r\n--%s--\r\n" % b).encode("utf8"))
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
data = open(src, "rb").read()

# 用一张小裁图（OCR 快一点）
import numpy as np
import cv2
bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
crop = bgr[:900, :]
ok, buf = cv2.imencode(".png", crop)
data = buf.tobytes()

t0 = time.time()
r = multipart({"title": "人工导入演示谱", "key": "D", "target": "C"},
              "manual_demo.png", data, B + "/api/import_manual")
print("提交:", r, "%.1fs" % (time.time() - t0))

last = ""
while True:
    jj = getj("/api/job")
    if jj.get("msg") != last:
        last = jj.get("msg")
        print("  ", last, "(%s/%s)" % (jj.get("done"), jj.get("total")))
    if not jj.get("running"):
        print("结束:", jj.get("error") or jj.get("msg"))
        break
    time.sleep(1.5)

# 校验结果
lib = getj("/api/lib")
hits = [x for x in lib["items"] if x.get("title") == "人工导入演示谱"]
print("导入结果：%d 页" % len(hits))
for h in hits:
    print("  id=%s key=%s target=%s status=%s no_ai=%s ocr=%s"
          % (h["id"][:8], h.get("key"), h.get("target"), h.get("status"),
             h.get("no_ai"), bool(h.get("ocr_txt"))))
    tp = h.get("ocr_txt")
    if tp and os.path.exists(tp):
        txt = open(tp, encoding="utf8").read()
        print("  OCR 文本 %d 字，前 40 字：%r" % (len(txt), txt[:40]))

# 打开看音符
if hits:
    post("/api/open", {"id": hits[0]["id"]})
    while True:
        jj = getj("/api/job")
        if not jj["running"]:
            break
        time.sleep(0.4)
    st = getj("/state")
    print("打开：音符 %d，橙框 %d"
          % (len([n for n in st["notes"] if not n.get("u")]),
             len([n for n in st["notes"] if n.get("u")])))
