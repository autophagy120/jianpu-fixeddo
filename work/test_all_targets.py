"""逐个目标调试：/target 后 /img 是否都能正常出图（找出会失败/不变的调）。"""
import hashlib
import json
import urllib.request

B = "http://127.0.0.1:8769"
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
        import time
        time.sleep(0.4)


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
print("源调:", getj("/state").get("source_key"))
seen = {}
for k in KEYS:
    try:
        post("/target", {"key": k})
        st = getj("/state")
        data = urllib.request.urlopen(B + "/img", timeout=600).read()
        h = hashlib.sha1(data).hexdigest()[:10]
        seen.setdefault(h, []).append(k)
        print("  1=%-3s state=%-3s 图 %d KB  %s" % (k, st.get("target_key"), len(data) // 1024, h))
    except Exception as e:
        print("  1=%-3s 失败: %s" % (k, e))
print("\n不同图像的种数:", len(seen), "（12 个调应该都不同）")
coll = {h: ks for h, ks in seen.items() if len(ks) > 1}
print("撞图的调:", coll)

# 还原成 C
post("/target", {"key": "C"})
print("已还原 1=C")
