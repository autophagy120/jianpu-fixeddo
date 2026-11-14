"""复现：框选一行做「加强识别」，看新增音符的行号/坐标是否错位。"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, r"D:\简谱转调项目")

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
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()

st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
rows = {}
for n in notes:
    rows.setdefault(n["row"], []).append(n)
print("页内行数:", len(rows), " 音符:", len(notes))

# 取中间一行做 ROI（左边界稍微往外扩一点，模拟"框一行"）
r = sorted(rows)[5]
line = sorted(rows[r], key=lambda n: n["x"])
x0 = min(n["x"] for n in line) - 20
x1 = max(n["x"] + n["w"] for n in line) + 20
y0 = min(n["y"] for n in line) - 24
y1 = max(n["y"] + n["h"] for n in line) + 24
roi = [x0, y0, x1 - x0, y1 - y0]
print("ROI: 行 %s  %d 个音  (%d,%d,%d,%d)" % (r, len(line), *[int(v) for v in roi]))
print("行基线/字高: base=%.1f  h=%.1f"
      % (sorted(n["y"] + n["h"] for n in line)[len(line) // 2],
         sorted(n["h"] for n in line)[len(line) // 2]))

before = {n["id"] for n in notes}
r0 = post("/strengthen", {"mode": "ocr", "roi": roi})
print("加强识别返回:", r0)
st2 = getj("/state")
notes2 = [n for n in st2["notes"] if not n.get("u")]
new = [n for n in notes2 if n["id"] not in before]
print("新增音符:", len(new))
for n in new[:10]:
    print("   #%s  x=%.0f y=%.0f w=%.0f h=%.0f row=%s digit=%s"
          % (n["id"], n["x"], n["y"], n["w"], n["h"], n["row"], n["digit"]))
same_row = [n for n in new if n["row"] == r]
print("落在本行(row=%s)的: %d / %d" % (r, len(same_row), len(new)))
others = [n for n in new if n["row"] != r]
if others:
    print("行号不对的示例:", [(n["row"], round(n["y"]), n["digit"]) for n in others[:8]])
