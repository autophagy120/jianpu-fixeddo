"""端到端：用无头浏览器打开 /read，检查布局与预渲染后是否 0 等待翻页。"""
import json
import subprocess
import time
import urllib.request

B = "http://127.0.0.1:8769"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

lib = json.loads(urllib.request.urlopen(B + "/api/lib", timeout=120).read())
groups = lib["groups"]
# 3 页、4 页、5 页各拿一份
from collections import Counter, defaultdict
by = defaultdict(list)
for x in lib["items"]:
    if x.get("group"):
        by[x["group"]].append(x)
targets = {}
for gid, pages in by.items():
    n = len(pages)
    if n in (3, 4, 5) and n not in targets:
        targets[n] = sorted(pages, key=lambda p: p.get("group_pos") or 0)

for n, pages in sorted(targets.items()):
    pid = pages[0]["id"]
    url = B + "/read?id=" + pid
    t0 = time.time()
    html = subprocess.run([EDGE, "--headless=new", "--disable-gpu",
                           "--virtual-time-budget=25000", "--dump-dom", url],
                          capture_output=True, text=True, encoding="utf8",
                          errors="replace", timeout=120).stdout
    dur = time.time() - t0
    ok_js = "预渲染" in html or "pageinfo" in html
    print("%d 页份：%s  用时 %.1fs  DOM %d 字  含预渲染/页码: %s"
          % (n, groups[pages[0]["group"]]["name"][:12], dur, len(html), ok_js))
