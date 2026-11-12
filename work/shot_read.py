"""截图检查阅读布局（3 页一份、4 页一份）。"""
import json
import os
import subprocess
import time
import urllib.request

B = "http://127.0.0.1:8769"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = r"D:\简谱转调项目\out"

lib = json.loads(urllib.request.urlopen(B + "/api/lib", timeout=120).read())
from collections import defaultdict
by = defaultdict(list)
for x in lib["items"]:
    if x.get("group"):
        by[x["group"]].append(x)

picked = {}
for gid, pages in by.items():
    n = len(pages)
    if n in (3, 4) and n not in picked:
        picked[n] = sorted(pages, key=lambda p: p.get("group_pos") or 0)

for n, pages in picked.items():
    url = B + "/read?id=" + pages[0]["id"]
    png = os.path.join(OUT, "_read_%d页.png" % n)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--window-size=1600,900",
                    "--virtual-time-budget=30000", "--screenshot=" + png, url],
                   capture_output=True, timeout=180)
    print("%d 页份截图: %s (%d KB)" % (n, png, os.path.getsize(png) // 1024 if os.path.exists(png) else -1))
