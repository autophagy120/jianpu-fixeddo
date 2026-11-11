import sys
sys.path.insert(0, ".")
import numpy as np
from jp import core
from jp.refimg import relighting, deskew

p = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
g = core.load_gray(p)
r = relighting.adaptive_lighting_correction(g)
print("relight 返回类型:", type(r), len(r) if isinstance(r, tuple) else "")
rel = r[0] if isinstance(r, tuple) else r
rel = np.asarray(rel)
print("  dtype", rel.dtype, "range", float(rel.min()), float(rel.max()))
rel8 = np.clip(rel * 255 if rel.max() <= 1.0 else rel, 0, 255).astype(np.uint8)
core.imwrite("out/ref_relight.png", rel8)

d = deskew.deskew_image(rel8)
print("deskew 返回类型:", type(d), len(d) if isinstance(d, tuple) else "")
dd = d[0] if isinstance(d, tuple) else d
dd = np.asarray(dd)
print("  dtype", dd.dtype, "range", float(dd.min()), float(dd.max()))
d8 = np.clip(dd * 255 if dd.max() <= 1.0 else dd, 0, 255).astype(np.uint8)
core.imwrite("out/ref_deskew.png", d8)
print("写出 out/ref_relight.png out/ref_deskew.png")
