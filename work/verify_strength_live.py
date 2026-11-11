"""验证三档识别强度（线上）：纯蓝 严格/标准/加强 的音符与橙框数。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


for _ in range(60):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

it = [x for x in j["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        break
    time.sleep(0.4)

for level, name in ((0, "严格"), (1, "标准"), (2, "加强")):
    t0 = time.time()
    r = post("/strength", {"level": level})
    print("强度 %s：音符 %3d  橙框 %3d  (%.1fs)  ok=%s"
          % (name, r.get("pairs", -1), r.get("uncovered", -1), time.time() - t0, r.get("ok")))
    time.sleep(0.3)

st = getj("/state")
print("当前档位:", st.get("strength"), " 音符:", len([n for n in st["notes"] if not n.get("u")]))
h = urllib.request.urlopen(B + "/", timeout=30).read().decode("utf8")
print("页面含识别强度下拉:", 'id="stg"' in h and "setStrength" in h)
