"""验证：对"整行都没识别"的区域做加强识别，新音应当留在本行（不被画到别的行）。"""
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
# 找橙框最多的页
best = None
for x in lib["items"]:
    if not x.get("ocr_txt"):
        continue
    post("/api/open", {"id": x["id"]})
    wait_job()
    st = getj("/state")
    unc = [n for n in st["notes"] if n.get("u")]
    if not unc:
        continue
    if best is None or len(unc) > best[1]:
        best = (x["id"], len(unc), [n for n in st["notes"] if not n.get("u")], unc)
    if len(unc) >= 60:
        break

pid, n_unc, notes, unc = best
print("测试页 %s：音符 %d，橙框 %d" % (pid[:8], len(notes), n_unc))

# 取一个"附近没有已识别音符"的橙框
target = None
for u in unc:
    cy = u["y"] + u["h"] / 2
    near = [n for n in notes if abs((n["y"] + n["h"] / 2) - cy) < 25 and abs(n["x"] - u["x"]) < 400]
    if len(near) <= 2:
        target = u
        break
if target is None:
    print("没找到孤立橙框，退出")
    raise SystemExit(0)

cy = target["y"] + target["h"] / 2
same_line = [u for u in unc if abs((u["y"] + u["h"] / 2) - cy) < 25]
x0 = min(u["x"] for u in same_line) - 20
x1 = max(u["x"] + u["w"] for u in same_line) + 20
roi = [x0, cy - 40, x1 - x0, 80]
print("ROI: (%d,%d,%d,%d)  该行橙框 %d 个" % (*[int(v) for v in roi], len(same_line)))

before = {n["id"] for n in notes}
r = post("/strengthen", {"mode": "ocr", "roi": roi})
print("加强识别:", r)
st2 = getj("/state")
notes2 = [n for n in st2["notes"] if not n.get("u")]
new = [n for n in notes2 if n["id"] not in before]
print("新增 %d 个；它们的 row / y：" % len(new))
for n in new[:12]:
    dy = (n["y"] + n["h"] / 2) - cy
    print("   row=%-6s y=%.0f (相对该行中心 %+.0f)  digit=%s" % (n["row"], n["y"], dy, n["digit"]))
far = [n for n in new if abs((n["y"] + n["h"] / 2) - cy) > 40]
print("明显跑到别的行的:", len(far), "/", len(new))
