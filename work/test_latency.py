"""量线上实际耗时：点击一次要等多久（POST + 拉图 + 拉状态）。"""
import json, time, urllib.request

B = "http://127.0.0.1:8769"


def get(p):
    return urllib.request.urlopen(B + p, timeout=120).read()


def post(p, o):
    t = time.time()
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    body = urllib.request.urlopen(r, timeout=300).read()
    return time.time() - t, json.loads(body)


for _ in range(90):
    try:
        st = json.loads(get("/state"))
        break
    except Exception:
        time.sleep(2)

notes = [n for n in st["notes"] if not n.get("u")]
n0 = notes[30]
print("音符 #%d  %s 点%d" % (n0["id"], n0["digit"], n0["oct"]))


def timed(label, fn, n=3):
    ts = []
    for _ in range(n):
        ts.append(fn())
    ts.sort()
    print("%-28s %.2fs  (min %.2f / max %.2f)" % (label, ts[len(ts) // 2], ts[0], ts[-1]))
    return ts


timed("GET /state（不重画）", lambda: (lambda t0: (get("/state"), time.time() - t0)[1])(time.time()))
timed("GET /img（1.4MB 图）", lambda: (lambda t0: (get("/img"), time.time() - t0)[1])(time.time()))


def do_edit():
    t1 = time.time()
    post("/edit", {"id": n0["id"], "digit": n0["digit"], "oct": n0["oct"]})
    get("/img")
    get("/state")
    return time.time() - t1


timed("点一下数字（含拉图）", do_edit)


def do_size():
    t1 = time.time()
    post("/size", {"ids": [n0["id"]], "cls": "aux"})
    get("/img")
    post("/size", {"ids": [n0["id"]], "cls": "main"})
    get("/img")
    return time.time() - t1


timed("改字号（来+回 2 次重画）", do_size, n=2)
print("回退:", post("/undo", {})[1], " 再回退:", post("/undo", {})[1])
