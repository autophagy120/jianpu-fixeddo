"""测试多格式导入：gif 原样存、avif 自动转 jpg；并验证来源仍可处理。"""
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jp.library as L

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=120).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def upload(path, name, sid=""):
    bnd = "----jp" + uuid.uuid4().hex
    data = open(path, "rb").read()
    body = b""
    if sid:
        body += ("--%s\r\nContent-Disposition: form-data; name=\"source_id\"\r\n\r\n%s\r\n"
                 % (bnd, sid)).encode()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"files\"; filename=\"%s\"\r\n"
             "Content-Type: application/octet-stream\r\n\r\n" % (bnd, name)).encode()
    body += data + ("\r\n--%s--\r\n" % bnd).encode()
    req = urllib.request.Request(B + "/api/upload", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


for _ in range(60):
    try:
        getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

src = post("/api/source_add", {"name": "格式测试", "kind": "upload"})["source"]["id"]
before = len(getj("/api/lib")["items"])

for f, name in ((r"work\_test.gif", "测试GIF/a.gif"), (r"work\_test.avif", "测试AVIF/b.avif")):
    if not os.path.exists(f):
        print("缺少测试文件", f)
        continue
    r = upload(f, name, src)
    print("%-6s 上传 -> %s" % (os.path.splitext(f)[1], r))

j = getj("/api/lib")
new = [x for x in j["items"] if x.get("source_id") == src]
print("库条目 %d -> %d，新增 %d 个" % (before, len(j["items"]), len(new)))
for x in new:
    p = x["image"]
    print("  %-28s 存成 %-9s 存在=%s 大小=%.0fKB" %
          (os.path.basename(x["origin"]), os.path.basename(p), os.path.exists(p),
           os.path.getsize(p) / 1024 if os.path.exists(p) else 0))

# 这两种格式都要能被几何分析读入（真跑一遍 core.analyse）
from jp import core
for x in new:
    try:
        page = core.analyse(x["image"])[0]
        print("  core.analyse OK:", os.path.basename(x["image"]), page.gray.shape)
    except Exception as e:
        print("  core.analyse 失败:", os.path.basename(x["image"]), e)

# 清理测试条目
d = L.load()
for k, v in list(d.items()):
    if v.get("source_id") == src:
        import shutil
        shutil.rmtree(L.store_dir(k), ignore_errors=True)
        del d[k]
L.save(d)
post("/api/source_del", {"id": src})
print("清理后条目:", len(L.load()), " 导入源:", len(getj("/api/sources")["sources"]))
