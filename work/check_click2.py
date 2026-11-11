import json
import time
import urllib.request

B = "http://127.0.0.1:8769"
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

r = urllib.request.urlopen(B + "/", timeout=60)
h = r.read().decode("utf8")
print("页面长度:", len(h))
for k in ("drag.w*scale>10", "log('click (", "if(n){ showSel(n, d.ctrl); }",
          "hideIns();", "showIns(p.x,p.y);"):
    print("  %-26s %s" % (k, k in h))
print("Cache-Control:", r.headers.get("Cache-Control"))
st = json.loads(urllib.request.urlopen(B + "/state", timeout=120).read())
print("音符:", len([n for n in st["notes"] if not n.get("u")]),
      " 橙框:", len([n for n in st["notes"] if n.get("u")]))
