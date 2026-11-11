"""用"标题唯一"的页做端到端持久化测试，避免同名声明的干扰。"""
import json
import time
import urllib.request
from collections import Counter

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def wait_job():
    while True:
        j = getj("/api/job")
        if not j["running"]:
            return j
        time.sleep(0.4)


def first_digit():
    st = getj("/state")
    ns = [n for n in st["notes"] if not n.get("u")]
    return (ns[0]["digit"] if ns else None), st.get("autosaved"), len(ns)


lib = getj("/api/lib")
cnt = Counter((x.get("title") or "") for x in lib["items"])
uniq = [x for x in lib["items"] if cnt.get(x.get("title") or "") == 1
        and (x.get("title") or "") and (x.get("ocr_txt") or "")]
print("标题唯一的页:", len(uniq))
target = uniq[0]
other = [x for x in lib["items"] if x["id"] != target["id"]][0]
print("测试页:", target["id"][:8], target.get("title"))

st0 = None
post("/api/open", {"id": target["id"]})
print("打开:", wait_job().get("error") or "ok")
d0, _, n0 = first_digit()
print("改前:", d0, "音符", n0)
newd = "7" if d0 != "7" else "3"
post("/edit", {"id": 0, "digit": newd, "oct": 0})
print("改成:", first_digit()[0])
time.sleep(3.0)
print("3 秒后（已自动保存）:", first_digit())

post("/api/open", {"id": other["id"]})
wait_job()
post("/api/open", {"id": target["id"]})
print("重开:", wait_job().get("error") or "ok")
d1, saved, n1 = first_digit()
print("重开后:", d1, "autosaved=", saved, " 音符", n1)
print("持久化成功:", d1 == newd)
