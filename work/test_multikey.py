"""多原调：detect_keys 找所有 1=X；_segments_for 按位置切段。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from jp import core, library as LIB, music as M

ok = True


def chk(name, got, exp):
    global ok
    good = got == exp
    ok = ok and good
    print(("PASS " if good else "FAIL ") + name)
    if not good:
        print("   got     ", got)
        print("   expect  ", exp)


txt = "\n".join([
    "简谱 1=G 4/4",            # 0
    "6 6 6 5 3 2",             # 1
    "",                         # 2
    "转1=D",                    # 3
    "1 2 3 4 5",               # 4
    "",                         # 5
    "1=D",                      # 6  重复，不该再多记
])
chk("多调识别（按行号）", [(k["key"], k["line"]) for k in LIB.detect_keys(txt)],
    [("G", 0), ("D", 3)])

chk("单调识别", [(k["key"], k["line"]) for k in LIB.detect_keys("1=bA\n0 0 0")],
    [("G#", 0)])

chk("无调号返回空", LIB.detect_keys("啦啦啦\n"), [])

# 切段：页高 1000，OCR 10 行，第 5 行转 D → 分界 y=500
page = SimpleNamespace(gray=SimpleNamespace(shape=(1000, 800)))
pairs = []
for i, y in enumerate([100, 300, 700, 900]):
    pairs.append((core.Note(digit="6", x=i * 50, box=(i * 50, y, i * 50 + 20, y + 30)), None))
keys = [dict(key="G", line=0), dict(key="D", line=5)]
txt2 = "\n".join(["x"] * 10)
segs = LIB._segments_for(page, pairs, keys, txt2)
G, D = M.parse_key("G"), M.parse_key("D")
chk("按位置切段（500 以下换调）", segs, [G, G, D, D])

# 只给一个调 → 不切段（返回空列表表示全谱一个调）
chk("单调不切段", LIB._segments_for(page, pairs, [dict(key="G", line=0)], txt2), [])

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
