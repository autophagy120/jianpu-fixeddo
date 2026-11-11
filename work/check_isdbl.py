import json
import time
import urllib.request

B = "http://127.0.0.1:8769"
v = json.loads(urllib.request.urlopen(B + "/api/version", timeout=60).read())
print("build:", v["build"])
h = urllib.request.urlopen(B + "/", timeout=60).read().decode("utf8")
print("页面还含 isDbl:", "isDbl" in h)
print("含 click 日志:", "log('click (" in h)
print("含 小位移规则:", "_small" in h)
print("页面长度:", len(h))

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
