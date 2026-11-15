"""验证：自动保存不会把"撤销"后的状态覆盖回去（含撤销前后都等自动保存的情况）。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=600).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def wait_job():
    while True:
        j = getj("/api/job")
        if not j["running"]:
            return j
        time.sleep(0.4)


def first_digit():
    ns = [n for n in getj("/state")["notes"] if not n.get("u")]
    return str(ns[0]["digit"]), int(ns[0]["oct"])


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
pid = it["id"]
post("/api/open", {"id": pid})
wait_job()
A = first_digit()
print("初始第一音:", A)

# 场景 1：改完立刻撤销（自动保存还没落盘）
B1 = "7" if A[0] != "7" else "2"
post("/edit", {"id": 0, "digit": B1, "oct": 0})
time.sleep(0.2)
print("撤销:", post("/undo", {}))
time.sleep(3.2)                       # 等自动保存
post("/api/open", {"id": pid})        # 重开
wait_job()
print("场景1 改完立刻撤销 → 重开:", first_digit(), "（应为", A, "）")

# 场景 2：改完等保存，再撤销，再等保存
post("/edit", {"id": 0, "digit": B1, "oct": 0})
time.sleep(3.2)
print("已保存改后状态，第一音:", first_digit())
print("撤销:", post("/undo", {}))
time.sleep(3.2)
post("/api/open", {"id": pid})
wait_job()
print("场景2 保存后撤销 → 重开:", first_digit(), "（应为", A, "）")

# 还原
post("/edit", {"id": 0, "digit": A[0], "oct": A[1]})
time.sleep(2.2)
print("已还原:", first_digit())
