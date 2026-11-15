"""验证目标调持久化：改 → 关掉重开 → 还是那个调。"""
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


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
pid = it["id"]
post("/api/open", {"id": pid})
wait_job()
print("打开：target=", getj("/state").get("target_key"), " 库里 target=", (json.loads(urllib.request.urlopen(B + "/api/lib").read()))["items"][0].get("target"))

post("/target", {"key": "G"})
time.sleep(0.5)
st = getj("/state")
lib2 = getj("/api/lib")
me = [x for x in lib2["items"] if x["id"] == pid][0]
print("改 1=G → state.target=", st.get("target_key"), " 库里 target=", me.get("target"))

# 模拟"关掉再打开"
post("/api/open", {"id": pid})
wait_job()
st2 = getj("/state")
print("重开后 target=", st2.get("target_key"), "（应为 G）→", st2.get("target_key") == "G")

# 还原成 C
post("/target", {"key": "C"})
time.sleep(0.4)
print("已还原 1=C，库里 target=", [x for x in getj("/api/lib")["items"] if x["id"] == pid][0].get("target"))
