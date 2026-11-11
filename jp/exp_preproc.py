"""测：接上参考项目的预处理后，三月雨/ BadApple 的识别结果怎么变。"""
import sys
sys.path.insert(0, ".")
import numpy as np

from jp import core, preproc, compare
from jp.glyphnet import GlyphNet

net = GlyphNet()
CASES = [
    (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg", "三月雨 742px"),
    (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\Bad Apple！！\371057593e3fc10d8bb [原始大小].jpg", "BadApple 2480px"),
]


def report(tag, gray):
    pg = core.analyse(gray, net, path=tag)
    tot = sum(len(s.notes) for s in pg.systems)
    low = sum(1 for s in pg.systems for n in s.notes if n.conf < 0.6)
    first = ""
    m = [s for s in pg.systems if s.notes]
    if m:
        first = " ".join(compare.token_of(n) for n in m[0].notes[:18])
    print(f"  {tag:16s} 行={len(pg.systems):2d} 音符={tot:3d} 低置信={low:3d} scale={pg.scale:.0f}")
    print(f"      首行: {first}")
    return tot, low


for path, name in CASES:
    g = core.load_gray(path)
    print(f"\n===== {name} =====")
    report("原图", g)
    r, info1 = preproc.run(g, do_relight=True, do_deskew=False)
    print("   relight info:", info1)
    report("relight", r)
    rd, info2 = preproc.run(g, do_relight=True, do_deskew=True)
    print("   +deskew info:", info2)
    report("relight+deskew", rd)
