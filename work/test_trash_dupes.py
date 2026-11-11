"""垃圾箱 + 重复检测 + origin 修复 的端到端验证。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"
FOLDER = r"C:\Users\sxm3q\Pictures\保存简谱"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


for _ in range(90):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

# 0) 重扫，修复 origin
r = post("/api/rescan", {})
j = getj("/api/lib")
items = [x for x in j["items"] if FOLDER.lower() in (x.get("origin") or "").lower()]
print("重扫:", r, " 该文件夹条目:", len(items), " 库共:", len(j["items"]), " 垃圾箱:", j.get("trash_n"))
import collections, os
print("  origin 扩展名:", dict(collections.Counter(os.path.splitext(x["origin"])[1].lower() for x in items)))
print("  落库扩展名:", dict(collections.Counter(os.path.splitext(x["image"])[1].lower() for x in items)))
if items:
    print("  样例:", items[0]["origin"].split("\\")[-1][:46], "→", items[0]["image"].split("\\")[-1])

# 1) 删除 → 垃圾箱 → 重扫不复活
victim = items[0]["id"]
print("\n删除:", post("/api/delete", {"ids": [victim]}))
j = getj("/api/lib")
print("  库:", len(j["items"]), " 垃圾箱:", j.get("trash_n"))
r2 = post("/api/rescan", {})
j2 = getj("/api/lib")
print("  重扫后仍在垃圾箱:", not any(x["id"] == victim for x in j2["items"]), " 垃圾箱:", j2.get("trash_n"))

# 2) 恢复
print("恢复:", post("/api/restore", {"ids": [victim]}))
print("  恢复后库里有:", any(x["id"] == victim for x in getj("/api/lib")["items"]))

# 3) 重复检测
print("\n重复检测:", post("/api/dupes", {}))
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        print("  ", jj["error"] or jj["msg"])
        break
    time.sleep(0.5)
j4 = getj("/api/lib")
dups = [x for x in j4["items"] if x.get("dup_of")]
print("疑似重复 %d 页：" % len(dups))
for x in dups[:8]:
    print("   ", (x.get("origin") or x["image"]).split("\\")[-1][:52], " ≈ ", x["dup_of"])
