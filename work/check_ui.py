import json
import time
import urllib.request

B = "http://127.0.0.1:8769"
for _ in range(60):
    try:
        j = json.loads(urllib.request.urlopen(B + "/api/lib", timeout=60).read())
        break
    except Exception:
        time.sleep(2)
print("页:", len(j["items"]), "组:", len(j.get("groups") or {}))
h = urllib.request.urlopen(B + "/lib", timeout=30).read().decode("utf8")
for k in ('g.collapsed===false', 'id="left"', 'id="right"', "pvShow", "group_new", "grpDrop"):
    print("  含 %-22s %s" % (k, k in h))
