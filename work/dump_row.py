"""核对：目标调 C 下，行 6 每个音应当渲染成什么（原调 G vs 中点起转 D）。"""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jp import music as M

B = "http://127.0.0.1:8769"
st = json.loads(urllib.request.urlopen(B + "/state", timeout=60).read())
notes = [n for n in st["notes"] if not n.get("u")]
row = 6
ns = sorted([n for n in notes if n["row"] == row], key=lambda n: n["x"])
anchor_x = 1166.0
ref = M.parse_key("C")
G = M.parse_key("G")
D = M.parse_key("D")
harm = M.HarmonicaOptions()

print("行%d 共 %d 个音，目标调 C（1=C）" % (row, len(ns)))
print("%5s %6s %4s %4s | %-8s | %-8s" % ("x", "deg", "acc", "oct", "全谱G", "中点后D"))
for n in ns:
    deg, acc, octv = int(n["digit"]), "", int(n["oct"])
    cx = n["x"] + n["w"] / 2.0
    outs = []
    for key in (G, D if cx >= anchor_x else G):
        p = M.note_to_abs(deg, acc, octv, key)
        tok = M.render_fixed_do([p], ref, harm, "cn", octave_bias=0)[0]
        outs.append(tok)
    mark = "  <-- 转调点" if abs(cx - anchor_x) < max(6.0, n["w"]) else ""
    print("%5.0f %6d %4s %4d | %-8s | %-8s%s" % (n["x"], deg, acc or "-", octv,
                                                outs[0], outs[1], mark))
