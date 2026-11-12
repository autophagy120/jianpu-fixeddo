"""5 页一份的布局截图（2×2 + 最后 1 页）+ 检查翻页切换是否本地即时。"""
import json
import os
import subprocess
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
five = [(g, p) for g, p in by.items() if len(p) == 5]
gid, pages = five[0]
pages.sort(key=lambda p: p.get("group_pos") or 0)
url = B + "/read?id=" + pages[0]["id"]
png = os.path.join(OUT, "_read_5页.png")
subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--window-size=1600,900",
                "--virtual-time-budget=40000", "--screenshot=" + png, url],
               capture_output=True, timeout=240)
print("5 页份:", lib["groups"][gid]["name"], "截图", os.path.getsize(png) // 1024, "KB")
