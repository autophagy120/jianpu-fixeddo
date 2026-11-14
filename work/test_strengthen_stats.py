"""统计：加强识别新增的音，是否落在框选区域内（不该跑到别的行）。"""
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
pages = [x for x in lib["items"] if x.get("ocr_txt")][:8]
tot_new = tot_far = 0
examples = []
for x in pages:
    post("/api/open", {"id": x["id"]})
    wait_job()
    st = getj("/state")
    notes = [n for n in st["notes"] if not n.get("u")]
    unc = [n for n in st["notes"] if n.get("u")]
    if not unc:
        continue
    # 取 3 个区域的橙框簇做 ROI
    done = 0
    for u in unc:
        if done >= 3:
            break
        cy = u["y"] + u["h"] / 2
        same = [v for v in unc if abs((v["y"] + v["h"] / 2) - cy) < 30
                and abs(v["x"] - u["x"]) < 700]
        if len(same) < 3:
            continue
        x0 = min(v["x"] for v in same) - 15
        x1 = max(v["x"] + v["w"] for v in same) + 15
        roi = [x0, cy - 45, x1 - x0, 90]
        before = {n["id"] for n in notes}
        r = post("/strengthen", {"mode": "ocr", "roi": roi})
        st2 = getj("/state")
        notes2 = [n for n in st2["notes"] if not n.get("u")]
        new = [n for n in notes2 if n["id"] not in before]
        if new:
            done += 1
            tot_new += len(new)
            far = [n for n in new
                   if abs((n["y"] + n["h"] / 2) - cy) > 60 or n["x"] < roi[0] - 40
                   or n["x"] > roi[0] + roi[2] + 40]
            tot_far += len(far)
            if far:
                examples.append((x["id"][:8], x.get("title") or "", len(new), len(far),
                                 [(n["row"], round(n["y"]), n["digit"]) for n in far[:3]]))
        time.sleep(0.2)

print("新增音符合计:", tot_new, " 跑出框外的:", tot_far)
for e in examples[:6]:
    print("  页 %s %s 新增 %d，跑出 %d：%s" % e)
