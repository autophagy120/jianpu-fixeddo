"""分步验证：改音 → 看内存态 → 等自动保存 → 直接读磁盘 cache.pkl。"""
import glob
import json
import pickle
import sys
import time
import urllib.request

sys.path.insert(0, r"D:\简谱转调项目")
B = "http://127.0.0.1:8769"


def getj(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=300).read())


def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def first_digit():
    st = getj("/state")
    ns = [n for n in st["notes"] if not n.get("u")]
    return ns[0]["digit"], st.get("autosaved"), len(ns)


print("当前:", first_digit())
post("/edit", {"id": 0, "digit": "7", "oct": 0})
print("改后立刻:", first_digit())
time.sleep(3.5)
print("等 3.5s 后:", first_digit())

f = glob.glob(r"D:\简谱转调项目\library\37d477b5*\cache.pkl")[0]
c = pickle.load(open(f, "rb"))
print("磁盘 cache: 第 1 音 =", str(c["pairs"][0][1].digit),
      " mtime =", time.strftime("%H:%M:%S", time.localtime(__import__("os").path.getmtime(f))))
