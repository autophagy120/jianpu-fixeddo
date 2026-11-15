"""复现并验证：先改数字为 5，再点"低1"，数字必须还是 5。"""
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
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
notes = [n for n in getj("/state")["notes"] if not n.get("u")]
n0 = notes[0]
print("初始: #%s digit=%s oct=%s" % (n0["id"], n0["digit"], n0["oct"]))

post("/edit", {"id": n0["id"], "digit": "5", "oct": 0})
time.sleep(0.4)
cur = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("改成 5 后: digit=%s oct=%s" % (cur["digit"], cur["oct"]))

r = post("/oct", {"id": n0["id"], "oct": -1})
time.sleep(0.4)
cur2 = [n for n in getj("/state")["notes"] if not n.get("u")][0]
print("点低1 后: digit=%s oct=%s  （/oct ok=%s）" % (cur2["digit"], cur2["oct"], r.get("ok")))
print("数字保持不变:", cur2["digit"] == "5", " 八度已变:", cur2["oct"] == -1)

# 还原
post("/edit", {"id": n0["id"], "digit": n0["digit"], "oct": n0["oct"]})
time.sleep(0.4)
print("已还原为 digit=%s oct=%s" % (n0["digit"], n0["oct"]))
