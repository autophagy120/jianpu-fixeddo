import json
import urllib.request

B = "http://127.0.0.1:8769"
v = json.loads(urllib.request.urlopen(B + "/api/version", timeout=60).read())
h = urllib.request.urlopen(B + "/", timeout=60).read().decode("utf8")
print("服务 build:", v["build"], " page_len:", len(h))
for k in ("log('click (", "_small", "drag.w*scale>10", "showSel(n, d.ctrl)",
          "页面已加载 build", "id=\"log\""):
    print("  %-24s %s" % (k, k in h))
i = h.find("if(!d.moved")
print("---- mouseup 片段 ----")
print(h[i - 120:i + 460])
