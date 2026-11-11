import json
import time
import urllib.request

B = "http://127.0.0.1:8769"
r = urllib.request.urlopen(B + "/", timeout=60)
h = r.read().decode("utf8")
print("Cache-Control:", r.headers.get("Cache-Control"))
print("拖拽阈值 10px:", "drag.w*scale>10" in h)
print("点击日志:", "log('click (" in h)

j = json.loads(urllib.request.urlopen(B + "/api/lib", timeout=120).read())
it = [x for x in j["items"] if x.get("title") == "纯蓝"][0]
req = urllib.request.Request(B + "/api/open", data=json.dumps({"id": it["id"]}).encode(),
                             headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=600).read()
while True:
    jj = json.loads(urllib.request.urlopen(B + "/api/job", timeout=30).read())
    if not jj["running"]:
        break
    time.sleep(0.5)
st = json.loads(urllib.request.urlopen(B + "/state", timeout=120).read())
print("已打开纯蓝，音符:", len([n for n in st["notes"] if not n.get("u")]))
