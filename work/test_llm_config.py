"""验证 设置→AI 配置：保存 / 读取（key 打码）/ 测试连接的失败提示 / 清理。"""
import json
import os
import time
import urllib.request

B = "http://127.0.0.1:8769"
CFG = os.path.join(r"D:\简谱转调项目", "work", "llm.json")


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=120).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())


for _ in range(60):
    try:
        getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

print("初始:", getj("/api/llm"))
had = os.path.exists(CFG)
backup = open(CFG, encoding="utf8").read() if had else None

r = post("/api/llm_save", {"base": "https://example.invalid/v1", "model": "test-model",
                           "key": "dummy-key-for-test"})
print("\n保存:", r)
c = getj("/api/llm")
print("读回:", c)
print("  key 未完整回传:", "dummy-key-for-test" not in json.dumps(c))
print("  打码显示:", c["key_mask"])

t = post("/api/llm_test", {})
print("\n测试连接（假地址，应失败并给出原因）:", t)

# 留空 key 保存 → 不应把原 key 清掉
r2 = post("/api/llm_save", {"base": "https://example.invalid/v1", "model": "test-model", "key": ""})
print("\n留空 key 再保存:", r2.get("ok"), " has_key:", r2["config"]["has_key"])

# 清理：恢复原状（这个会话本来没有配置）
if backup is None:
    if os.path.exists(CFG):
        os.remove(CFG)
    print("\n已删除测试用 llm.json，保持未配置状态")
else:
    open(CFG, "w", encoding="utf8").write(backup)
    print("\n已恢复原有 llm.json")

print("清理后:", getj("/api/llm"))
h = urllib.request.urlopen(B + "/lib", timeout=30).read().decode("utf8")
for k in ("AI 配置…", "llmOpen", "llmSave", "llmTest"):
    print("  页面含 %-10s %s" % (k, k in h))
