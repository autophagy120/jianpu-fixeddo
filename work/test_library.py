"""端到端跑一遍乐谱库：导入 → 打开 → 处理 → AI（未配置应报错）→ 日志。"""
import json, time, urllib.request

B = "http://127.0.0.1:8769"
CORPUS = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱"


def get(p):
    return urllib.request.urlopen(B + p, timeout=600).read()


def getj(p):
    return json.loads(get(p))


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


for _ in range(90):
    try:
        get("/lib")
        break
    except Exception:
        time.sleep(2)

print("乐谱库页面:", len(get("/lib")), "字节")
j = getj("/api/lib")
print("初始:", j["stats"], " OCR引擎:", j["ocr"], " AI:", j["llm"])

for sub in ("三月雨", "Bad Apple！！"):
    r = post("/api/import", {"root": CORPUS + "\\" + sub})
    print("导入 %s ->" % sub, r)

items = getj("/api/lib")["items"]
print("\n库内容:")
for it in items:
    print("  %-8s %s  原调=%s 目标=%s 状态=%s" % (it["id"][:6], it["song"], it["key"], it["target"], it["status"]))

# 打开第一首（三月雨 有 OCR 缓存，不需要 GPU）
tgt = [x for x in items if "三月雨" in x["song"]][0]
print("\n打开:", tgt["song"], post("/api/open", {"id": tgt["id"]}))
for _ in range(120):
    jj = getj("/api/job")
    if not jj["running"]:
        print("  任务结束:", jj)
        break
    time.sleep(1)
print("  GET / ->", len(get("/")), "字节")
st = getj("/state")
print("  已打开:", st["source_key"], "音符", len([n for n in st["notes"] if not n.get("u")]))

# 处理（重新出图）
print("\n处理:", post("/api/process", {"id": tgt["id"]}))
for _ in range(240):
    jj = getj("/api/job")
    if not jj["running"]:
        print("  ", jj["msg"][:120], " error=", jj["error"])
        break
    time.sleep(2)
it = [x for x in getj("/api/lib")["items"] if x["id"] == tgt["id"]][0]
print("  处理后: 状态=%s 音符=%d 漏=%d out=%s" % (it["status"], it["notes"], it["uncovered"], it["out"]))

# AI（未配置）
print("\nAI:", post("/api/ai_meta", {"id": tgt["id"]}))

print("\n日志尾部:")
for line in getj("/api/log")["lines"][-6:]:
    print("  " + line)
