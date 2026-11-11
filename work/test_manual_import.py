"""测人工导入：不跑 OCR/AI，进库后打开应该是"0 音符 + 满屏橙框"，可逐点回填。"""
import json
import os
import time
import urllib.request
import uuid

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

before = len(j["items"])
# 用现成的一张图当"人工导入"的素材
src = None
for x in j["items"]:
    if x.get("title") == "纯蓝":
        src = x["image"]
        break
assert src and os.path.exists(src)
data = open(src, "rb").read()

boundary = "----wb" + uuid.uuid4().hex
parts = []
fields = {"title": "人工导入测试谱", "key": "C", "target": "C"}


def field(k, v):
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                  % (boundary, k, v)).encode("utf8"))


for k, v in fields.items():
    field(k, v)
parts.append(("--%s\r\nContent-Disposition: form-data; name=\"files\"; filename=\"test_page.png\"\r\n"
              "Content-Type: image/png\r\n\r\n" % boundary).encode("utf8"))
parts.append(data)
parts.append(("\r\n--%s--\r\n" % boundary).encode("utf8"))
body = b"".join(parts)

req = urllib.request.Request(B + "/api/import_manual", data=body,
                             headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
r = json.loads(urllib.request.urlopen(req, timeout=600).read())
print("导入结果:", r)
if not r.get("ok"):
    raise SystemExit(1)

i = r["ids"][0]
it = getj("/api/lib")["items"]
me = [x for x in it if x["id"] == i][0]
print("条目：title=%s key=%s status=%s no_ai=%s group=%s"
      % (me.get("title"), me.get("key"), me.get("status"), me.get("no_ai"), me.get("group")))
print("库页数：%d → %d" % (before, len(it)))

post("/api/open", {"id": i})
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        break
    time.sleep(0.4)
st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
unc = [n for n in st["notes"] if n.get("u")]
print("打开后：音符 %d，橙框 %d（橙框就是可点回填的音符位置）" % (len(notes), len(unc)))

if unc:
    r2 = post("/assign", {"slot": 0, "digit": "5", "oct": 0})
    st2 = getj("/state")
    print("点第一个橙框回填 5 → ok=%s，现在音符 %d"
          % (r2.get("ok"), len([n for n in st2["notes"] if not n.get("u")])))
