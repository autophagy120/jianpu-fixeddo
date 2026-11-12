"""验证：橙框旁边已有同音时，默认回填应该是"新增"，而不是把原来的挪走。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
while True:
    j = getj("/api/job")
    if not j["running"]:
        break
    time.sleep(0.4)

st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
unc = [n for n in st["notes"] if n.get("u")]
print("起始：音符 %d，橙框 %d" % (len(notes), len(unc)))
if not unc:
    raise SystemExit("这页没橙框，换一页测")

# 找"右边紧邻有同数字"的橙框不容易，直接给橙框填一个已存在的数字（比如 5）
target_digit = notes[0]["digit"]
before = len(notes)
before_digit_count = sum(1 for n in notes if n["digit"] == target_digit)

r = post("/assign", {"slot": 0, "digit": target_digit, "oct": 0})
st2 = getj("/state")
notes2 = [n for n in st2["notes"] if not n.get("u")]
after_digit_count = sum(1 for n in notes2 if n["digit"] == target_digit)
print("回填数字 %s：音符 %d → %d（%+d）  moved=%s" %
      (target_digit, before, len(notes2), len(notes2) - before, r.get("moved")))
print("该数字出现次数：%d → %d" % (before_digit_count, after_digit_count))
print("默认新增（不是挪走）:", len(notes2) == before + 1 and after_digit_count == before_digit_count + 1)

# 再测 move=true 的显式路径
if unc:
    r2 = post("/assign", {"slot": 1, "digit": notes2[0]["digit"], "oct": 0, "move": True})
    st3 = getj("/state")
    n3 = len([n for n in st3["notes"] if not n.get("u")])
    print("显式 move：moved=%s，音符 %d → %d" % (r2.get("moved"), len(notes2), n3))
