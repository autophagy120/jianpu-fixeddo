"""线上回归：连续两次加强识别，框1里的音一个都不能少；删除音能撤销。"""
import json, time, urllib.request

B = "http://127.0.0.1:8769"


def get(p):
    return urllib.request.urlopen(B + p, timeout=180).read()


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


for _ in range(90):
    try:
        st = json.loads(get("/state"))
        break
    except Exception:
        time.sleep(2)

notes = [n for n in st["notes"] if not n.get("u")]
rows = {}
for n in notes:
    rows.setdefault(n["row"], []).append(n)
row = sorted(rows)[6]
ns = sorted(rows[row], key=lambda n: n["x"])


def box(a, b):
    g = ns[a:b]
    x0 = min(n["x"] for n in g) - 6
    x1 = max(n["x"] + n["w"] for n in g) + 6
    y0 = min(n["y"] for n in g) - 6
    y1 = max(n["y"] + n["h"] for n in g) + 6
    return [x0, y0, x1 - x0, y1 - y0]


b1, b2 = box(2, 6), box(14, 18)


def inside(b):
    s = json.loads(get("/state"))
    x, y, w, h = b
    return sum(1 for n in s["notes"] if not n.get("u")
               and n["x"] >= x - 2 and n["x"] + n["w"] <= x + w + 2
               and n["y"] >= y - 2 and n["y"] + n["h"] <= y + h + 2)


n0 = len(notes)
i0 = inside(b1)
r1 = post("/strengthen", {"mode": "ocr", "roi": b1})
n1 = len([n for n in json.loads(get("/state"))["notes"] if not n.get("u")])
i1 = inside(b1)
print("第1次加强识别:", r1)
print("  音符 %d -> %d   框1内 %d -> %d" % (n0, n1, i0, i1))

r2 = post("/strengthen", {"mode": "ocr", "roi": b2})
n2 = len([n for n in json.loads(get("/state"))["notes"] if not n.get("u")])
i2 = inside(b1)
print("第2次加强识别:", r2)
print("  音符 %d -> %d   框1内 %d（必须 >= %d）" % (n1, n2, i2, i0))

# 删除一个音再撤销
s = json.loads(get("/state"))
tid = [n for n in s["notes"] if not n.get("u")][5]["id"]
d = post("/del", {"id": tid})
n3 = len([n for n in json.loads(get("/state"))["notes"] if not n.get("u")])
u = post("/undo", {})
n4 = len([n for n in json.loads(get("/state"))["notes"] if not n.get("u")])
print("删除 #%d -> %s  音符 %d；撤销 -> %s  音符 %d" % (tid, d, n3, u, n4))

ok = n1 >= n0 and n2 >= n1 and i2 >= i0 and n3 == n2 - 1 and n4 == n2
print("RESULT:", "PASS" if ok else "FAIL")
