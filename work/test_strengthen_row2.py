"""再测：确保当前打开的页就是被测页，再对"整行都没识别"的区域做加强识别。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=600).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def wait_job():
    while True:
        j = getj("/api/job")
        if not j["running"]:
            return j
        time.sleep(0.4)


lib = getj("/api/lib")
cand = [x for x in lib["items"] if x.get("ocr_txt") and x.get("notes")][:20]
best = None
for x in cand:
    post("/api/open", {"id": x["id"]})
    wait_job()
    st = getj("/state")
    unc = [n for n in st["notes"] if n.get("u")]
    if unc and (best is None or len(unc) > best[1]):
        best = (x["id"], len(unc))
    if best and best[1] >= 40:
        break

pid = best[0]
post("/api/open", {"id": pid})
wait_job()
st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
unc = [n for n in st["notes"] if n.get("u")]
print("页 %s：音符 %d 橙框 %d" % (pid[:8], len(notes), len(unc)))

# 找"附近已识别音符 < 2"的橙框 → 整行缺失的区域
target = None
for u in unc:
    cy = u["y"] + u["h"] / 2
    near = [n for n in notes if abs((n["y"] + n["h"] / 2) - cy) < 25]
    if len(near) <= 1:
        target = u
        break
if not target:
    print("没有整行缺失的区域")
    raise SystemExit(0)

cy = target["y"] + target["h"] / 2
same = [u for u in unc if abs((u["y"] + u["h"] / 2) - cy) < 25]
x0 = min(u["x"] for u in same) - 20
x1 = max(u["x"] + u["w"] for u in same) + 20
roi = [x0, cy - 40, x1 - x0, 80]
print("整行缺失：该行橙框 %d 个，ROI=(%d,%d,%d,%d) 行中心 y=%.0f"
      % (len(same), *[int(v) for v in roi], cy))

before = {n["id"] for n in notes}
r = post("/strengthen", {"mode": "ocr", "roi": roi})
print("加强识别:", r)
st2 = getj("/state")
notes2 = [n for n in st2["notes"] if not n.get("u")]
new = [n for n in notes2 if n["id"] not in before]
print("新增 %d 个：" % len(new))
for n in new[:12]:
    dy = (n["y"] + n["h"] / 2) - cy
    print("   row=%-6s y=%-6.0f 中心偏差 %+6.0f  digit=%s" % (n["row"], n["y"], dy, n["digit"]))
far = [n for n in new if abs((n["y"] + n["h"] / 2) - cy) > 40]
print("跑到别的行的:", len(far), "/", len(new))
