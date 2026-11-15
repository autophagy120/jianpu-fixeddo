"""查：改目标调后，state 和渲染图有没有真的变。"""
import hashlib
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


def img_hash():
    d = urllib.request.urlopen(B + "/img", timeout=600).read()
    return hashlib.sha1(d).hexdigest()[:12], len(d)


lib = getj("/api/lib")
it = [x for x in lib["items"] if x.get("title") == "纯蓝"][0]
post("/api/open", {"id": it["id"]})
wait_job()
st = getj("/state")
print("初始: source=%s target=%s 音符=%d" % (st.get("source_key"), st.get("target_key"),
                                        len([n for n in st["notes"] if not n.get("u")])))
h0 = img_hash()
print("图哈希:", h0)

for k in ("G", "C", "F"):
    r = post("/target", {"key": k})
    time.sleep(0.6)
    st2 = getj("/state")
    h = img_hash()
    print("设为 1=%-2s → 返回 %s  state.target=%s  图哈希 %s  变了:%s"
          % (k, r, st2.get("target_key"), h[0], h[0] != h0[0]))
    h0 = h

# 下拉框可选值能不能被 parse_key 认
import sys
sys.path.insert(0, r"D:\简谱转调项目")
import jp.music as M
bad = [k for k in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
       if M.parse_key(k) is None]
print("下拉里 parse_key 认不出的:", bad)
