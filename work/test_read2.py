"""验证新阅读布局：3 页 / 4 页 / 5 页都应该"一屏最多两页"。"""
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

picked = {}
for gid, pages in by.items():
    n = len(pages)
    if n in (3, 4, 5) and n not in picked:
        picked[n] = (gid, sorted(pages, key=lambda p: p.get("group_pos") or 0))

for n, (gid, pages) in sorted(picked.items()):
    url = B + "/read?id=" + pages[0]["id"]
    png = os.path.join(OUT, "_read2_%d页.png" % n)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--window-size=1600,900",
                    "--virtual-time-budget=40000", "--screenshot=" + png, url],
                   capture_output=True, timeout=240)
    dom = subprocess.run([EDGE, "--headless=new", "--disable-gpu",
                          "--virtual-time-budget=40000", "--dump-dom", url],
                         capture_output=True, text=True, encoding="utf8",
                         errors="replace", timeout=240).stdout
    cells = dom.count('class="cell"')
    chips = dom.count('class="chip')
    print("%d 页份 %-14s 首屏 cell=%d  chips=%d  截图 %d KB"
          % (n, lib["groups"][gid]["name"][:14], cells, chips,
             (os.path.getsize(png) // 1024) if os.path.exists(png) else -1))
