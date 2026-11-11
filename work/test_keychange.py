"""转调切段的纯函数测试：不加载页面，直接构造 pairs 验证 keys_for_pairs。

用户要求：转调点按**位置**锚定（可从一行半路开始），不是按行号。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jp import core
from jp.roi_server import State


class Fake(State):
    def __init__(self, pairs, key_pc=9, med_h=20.0):
        self.pairs = pairs
        self.key_pc = key_pc
        self.med_h = med_h
        self.key_changes = []


def note(x, y, deg="6", octv=0):
    s = core.Note(digit=deg, x=float(x), box=(x, y, x + 10, y + 18), row=0)
    return (s, None)


def run(name, pairs, changes, expect):
    st = Fake(pairs, key_pc=9)          # 原调 A
    st.key_changes = changes
    got = st.keys_for_pairs()
    ok = got == expect
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        print("  got   ", got)
        print("  expect", expect)
    return ok


allok = True
# 一行 6 个音，y=200；转调点落在第 3 个音左侧 x=130（半路）
pairs = [note(10, 200), note(60, 200), note(130, 200), note(180, 200)]
allok &= run("半行起转（点在第3音上）", pairs,
             [dict(x=130, y=200, key_pc=7)],
             [9, 9, 7, 7])

# 点在两个音之间（x=100；音中心 65 和 135）→ 中心在点右侧的音才转
allok &= run("点在音与音之间", pairs,
             [dict(x=100, y=200, key_pc=7)],
             [9, 9, 7, 7])

# 点在整行最左边 → 整行转
allok &= run("整行转（点在行首）", pairs,
             [dict(x=0, y=200, key_pc=7)],
             [7, 7, 7, 7])

# 点在整行最右边之后 → 本行不转
allok &= run("点在行末之后（本行不转）", pairs,
             [dict(x=999, y=200, key_pc=7)],
             [9, 9, 9, 9])

# 跨行：第二行整行转
pairs2 = pairs + [note(10, 260), note(80, 260)]
allok &= run("下一行起转", pairs2,
             [dict(x=0, y=258, key_pc=7)],
             [9, 9, 9, 9, 7, 7])

# 多个转调点：A → G（第1行半路）→ D（第2行）
allok &= run("多段转调", pairs2,
             [dict(x=130, y=200, key_pc=7), dict(x=0, y=258, key_pc=2)],
             [9, 9, 7, 7, 2, 2])

# 乱序添加也要按阅读顺序生效
allok &= run("乱序添加", pairs2,
             [dict(x=0, y=258, key_pc=2), dict(x=130, y=200, key_pc=7)],
             [9, 9, 7, 7, 2, 2])

# 无 TAG → None（保持旧的"全谱同调"路径，逻辑零改动）
st = Fake(pairs)
allok &= run("无转调点返回 None", pairs, [], None)

print("ALL OK" if allok else "FAILED")
sys.exit(0 if allok else 1)
