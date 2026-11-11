import json
import time
import urllib.request

B = "http://127.0.0.1:8769"
v = json.loads(urllib.request.urlopen(B + "/api/version", timeout=60).read())
print("服务版本:", v)

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

roi = urllib.request.urlopen(B + "/", timeout=60).read().decode("utf8")
lib = urllib.request.urlopen(B + "/lib", timeout=60).read().decode("utf8")
for k in ("build " + v["build"], "id=\"buildTag\"", "（操作记录会出现在这里）",
          "页面已加载 build"):
    print("  ROI 含 %-22s %s" % (k, k in roi))
print("  LIB 含 build 标记:", ("build " + v["build"]) in lib)
print("  ROI 页面长度:", len(roi))
