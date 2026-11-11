"""线上验证 DeepSeek：配置、测试连接、单页 AI 补信息、批量 AI 分类。"""
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


for _ in range(60):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

print("1) 配置:", getj("/api/llm"))
print("2) 测试连接:", post("/api/llm_test", {}))

withocr = []
for it in j["items"]:
    import os
    tp = it.get("ocr_txt") or ""
    if tp and os.path.exists(tp):
        withocr.append(it)
print("\n3) 有 OCR 文本、可做 AI 的页:", len(withocr))
if withocr:
    it = withocr[0]
    print("   拿这页试 AI 补信息:", it.get("id"), it.get("song"), " 现有标题:", repr(it.get("title")))
    r = post("/api/ai_meta", {"id": it["id"]})
    print("   结果:", json.dumps(r.get("meta"), ensure_ascii=False),
          " 写回标题:", repr((r.get("item") or {}).get("title")))

print("\n4) 批量 AI 分类（只处理已有 OCR 文本的页）:", post("/api/classify", {}))
t0 = time.time()
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        print("   ", jj["error"] or jj["msg"], " %.1fs" % (time.time() - t0))
        break
    time.sleep(0.6)

j2 = getj("/api/lib")
have = [(x.get("song"), x.get("title")) for x in j2["items"] if x.get("title")]
print("\n5) 库里带标题的页:", len(have))
for s, t in have[:8]:
    print("   %-28s -> %s" % ((s or "")[:28], t))
