"""测试：上传导入（multipart）→ 文件复制进库 → 打开缓存加速 → 导入源管理。"""
import io
import json
import os
import sys
import time
import urllib.request
import uuid

B = "http://127.0.0.1:8769"
ROOT = r"D:\简谱转调项目"
SRC_IMG = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"


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
    for k, v in (("source_id", sid), ("key", "")):
        if v:
            body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                     % (bnd, k, v)).encode()
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

before = len(getj("/api/lib")["items"])
src = post("/api/source_add", {"name": "测试上传", "kind": "upload", "note": "自动化测试"})
sid = src["source"]["id"]
print("新建导入源:", sid)

r = upload(SRC_IMG, "歌曲A/封面.jpg", sid)
print("上传结果:", r)
j = getj("/api/lib")
new = [x for x in j["items"] if x.get("source_id") == sid]
print("库条目 %d -> %d，新条目 %d 个" % (before, len(j["items"]), len(new)))
if new:
    it = new[0]
    print("  图片落库路径:", it["image"])
    print("  文件存在:", os.path.exists(it["image"]),
          " 大小:", os.path.getsize(it["image"]) if os.path.exists(it["image"]) else 0)
    print("  歌名(取自文件夹):", it["song"])

ss = getj("/api/sources")["sources"]
print("导入源数量:", len(ss))
print("删除测试源:", post("/api/source_del", {"id": sid}))
print("删除后数量:", len(getj("/api/sources")["sources"]))

# 打开缓存：同一页开两次，第二次应显著更快
items = [x for x in j["items"] if "三月雨" in x["song"]]
if items:
    iid = items[0]["id"]
    def open_once():
        t = time.time()
        post("/api/open", {"id": iid})
        while True:
            jj = getj("/api/job")
            if not jj["running"]:
                if jj["error"]:
                    raise RuntimeError(jj["error"])
                return time.time() - t
            time.sleep(0.5)
    t1 = open_once()
    print("\n第一次打开: %.1fs" % t1)
    print("保存缓存:", post("/api/save", {}))
    t2 = open_once()
    print("第二次打开: %.2fs  （加速 %.1fx）" % (t2, t1 / max(t2, 0.01)))
    st = getj("/state")
    print("音符数:", len([n for n in st["notes"] if not n.get("u")]),
          " 原调:", st["source_key"], " 目标调:", st["target_key"])
