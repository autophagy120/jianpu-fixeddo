"""验证：①回填橙框不再变两个 ②批量删除 ③框内清空。"""
import json
import time
import urllib.request

B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


for _ in range(60):
    try:
        j = getj("/api/lib")
        break
    except Exception:
        time.sleep(2)

it = [x for x in j["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
while True:
    jj = getj("/api/job")
    if not jj["running"]:
        break
    time.sleep(0.4)

st = getj("/state")
notes = [n for n in st["notes"] if not n.get("u")]
unc = [n for n in st["notes"] if n.get("u")]
print("打开 纯蓝：音符 %d，橙框 %d" % (len(notes), len(unc)))

# ① 选一个橙框，回填成它右边最近的音符相同的数字，看总数是否只 +1（而不是 +2）
if unc:
    u = unc[0]
    # 找右侧最近的同数字音
    cand = [n for n in notes if n["digit"] and
            abs((n["y"] + n["h"] / 2) - (u["y"] + u["h"] / 2)) < 30 and n["x"] > u["x"]]
    cand.sort(key=lambda n: n["x"])
    digit = cand[0]["digit"] if cand else "6"
    before = len(notes)
    r = post("/assign", {"slot": 0, "digit": digit, "oct": 0})
    st2 = getj("/state")
    after = len([n for n in st2["notes"] if not n.get("u")])
    print("回填 slot0（digit=%s）: 音符 %d → %d（%+d）  %s"
          % (digit, before, after, after - before,
             "✅ 挪动而不是新增" if after - before <= 1 else "❌ 还是变两个了"))

# ② 批量删除：删掉前 3 个音
ids = [n["id"] for n in getj("/state")["notes"] if not n.get("u")][:3]
r = post("/del_many", {"ids": ids})
n2 = len([n for n in getj("/state")["notes"] if not n.get("u")])
print("批量删除 %s → 删了 %d 个，剩 %d 音符" % (ids, r.get("n"), n2))

# ③ 框内清空：框住前 60 个音所在区域
notes2 = [n for n in getj("/state")["notes"] if not n.get("u")]
if notes2:
    g = notes2[:60]
    roi = [min(n["x"] for n in g) - 5, min(n["y"] for n in g) - 5,
           max(n["x"] + n["w"] for n in g) - min(n["x"] for n in g) + 10,
           max(n["y"] + n["h"] for n in g) - min(n["y"] for n in g) + 10]
    r = post("/roi_clear_notes", {"roi": roi})
    n3 = len([n for n in getj("/state")["notes"] if not n.get("u")])
    print("框内清空：删除 %d 个，剩 %d 音符" % (r.get("n"), n3))
