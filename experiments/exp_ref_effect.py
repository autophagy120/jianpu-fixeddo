import sys
sys.path.insert(0, ".")
import numpy as np
from jp import core
from jp import compare
from jp.glyphnet import GlyphNet

net = GlyphNet()
p = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"

variants = [("原图", core.load_gray(p))]
for name in ("out/ref_relight.png", "out/ref_deskew.png"):
    try:
        variants.append((name.split("/")[-1], core.load_gray(name)))
    except Exception as e:
        print("skip", name, e)

for tag, gray in variants:
    pg = core.analyse(gray, net, path=tag)
    tot = sum(len(s.notes) for s in pg.systems)
    music = [s for s in pg.systems if s.notes]
    first = " ".join(compare.token_of(n) for n in music[0].notes[:20]) if music else ""
    low = sum(1 for s in pg.systems for n in s.notes if n.conf < 0.6)
    print(f"{tag:22s} 行={len(pg.systems):2d} 音符={tot:3d} 低置信={low:3d} scale={pg.scale:.0f}")
    print(f"   首行: {first}")
