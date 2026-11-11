"""转调锚点吸附测试：行首点击（哪怕 y 偏一点）必须从该行开头生效。

注意：转调是**从那一点开始一直到后面**都换调（不是只换一行），
所以期望值 = 该行前半用原调、其余全部新调。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jp import core
from jp.roi_server import State

G, D = 7, 2          # 原调 1=G，新调 1=D


class Fake(State):
    def __init__(self, pairs, key_pc=G, med_h=32.0):
        self.pairs = pairs
        self.key_pc = key_pc
        self.med_h = med_h
        self.key_changes = []


def note(x, y, row):
    return (core.Note(digit="6", x=float(x), box=(x, y, x + 20, y + 32), row=row), None)


rows = [note(x, 100, 1) for x in (100, 160, 220, 280)] + \
       [note(x, 200, 2) for x in (100, 160, 220, 280)]

ok = True


def run(name, change, expect):
    global ok
    st = Fake(rows)
    st.key_changes = [change]
    got = st.keys_for_pairs()
    good = got == expect
    ok = ok and good
    print(("PASS " if good else "FAIL ") + name)
    if not good:
        print("   got   ", got)
        print("   expect", expect)


run("行首点击（y 正中）", dict(x=40, y=116, key_pc=D), [D] * 8)
run("行首点击（y 偏下 25px）", dict(x=40, y=141, key_pc=D), [D] * 8)
run("行首点击（y 偏上 20px）", dict(x=40, y=96, key_pc=D), [D] * 8)
run("行首点击（y 偏下 40px，仍在吸附范围）", dict(x=40, y=156, key_pc=D), [D] * 8)
run("行中点击（第 2 个音之前）", dict(x=155, y=116, key_pc=D), [G, D, D, D, D, D, D, D])
run("第二行行首", dict(x=40, y=216, key_pc=D), [G, G, G, G, D, D, D, D])
run("两行之间（更靠近第二行）", dict(x=40, y=190, key_pc=D), [G, G, G, G, D, D, D, D])
run("点在第 1 行最右之后", dict(x=400, y=116, key_pc=D),
    [G, G, G, G, D, D, D, D])

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
