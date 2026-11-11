"""验证"整行统一字号 + 统一基线"：行内数字的墨迹底边必须落在同一条线上。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from jp import emit_fixed as EF
from jp.roi_server import State

st = State("BadApple")
page, pairs = st.page, st.pairs
med_h = float(np.median([max(1.0, s.box[3] - s.box[1]) for s, _o in pairs]))
bgr, stats = EF.render_fixed_image(page, pairs, st.key_pc, st.target_pc, mark_missed=False)
print("渲染完成 stats:", {k: v for k, v in stats.items() if k in ("drawn", "rest", "aux", "shrunk", "overflow")})

R, G, B = bgr[:, :, 2].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 0].astype(int)
red = (R > 150) & (G < 80) & (B < 80)
H, W = red.shape


def ink_bottom(cx, base, gh):
    """量数字本身的墨迹底边。

    y 窗口**必须掐在基线附近**：低八度的红点画在数字下方 ~0.10gh+r 处，
    窗口放太宽会把点当成笔画的底，量出来就是假的"没对齐"（第一版就踩了这个）。
    """
    x0 = max(0, int(cx - 0.55 * gh))
    x1 = min(W, int(cx + 0.55 * gh))
    y0 = max(0, int(base - 1.35 * gh))
    y1 = min(H, int(round(base + 0.06 * gh)))
    ys = np.nonzero(red[y0:y1, x0:x1].any(axis=1))[0]
    return (y0 + int(ys.max())) if len(ys) else None


def ink_h(cx, base, gh):
    x0 = max(0, int(cx - 0.55 * gh))
    x1 = min(W, int(cx + 0.55 * gh))
    y0 = max(0, int(base - 1.5 * gh))
    y1 = min(H, int(round(base + 0.06 * gh)))
    ys = np.nonzero(red[y0:y1, x0:x1].any(axis=1))[0]
    return (int(ys.max() - ys.min() + 1)) if len(ys) else 0


by_row = {}
for s, o in pairs:
    by_row.setdefault(int(s.row), []).append(s)

print("\n%4s %5s %6s %8s %8s %8s" % ("行", "音数", "中位字高", "框底极差", "旧基线极差", "新基线极差"))
bad_rows = 0
for row in sorted(by_row):
    ss = sorted(by_row[row], key=lambda s: s.box[0])
    if len(ss) < 5:
        continue
    hs = [max(1.0, s.box[3] - s.box[1]) for s in ss]
    gh = float(np.median(hs))
    gh = min(1.25 * med_h, max(0.80 * med_h, gh))
    bottoms = [float(s.box[3]) for s in ss]
    base = float(np.median(bottoms))

    # 旧规则：每个音各自按槽位框底边 -> 基线就是各自的 y1
    old_spread = max(bottoms) - min(bottoms)

    # 新规则：测量画出来的红字墨迹底边 / 字高
    vals = []
    hts = []
    for s in ss:
        cx = (s.box[0] + s.box[2]) / 2.0
        b = ink_bottom(cx, base, gh)
        if b is not None:
            vals.append(b)
            hts.append(ink_h(cx, base, gh))
    new_spread = (max(vals) - min(vals)) if vals else -1
    h_spread = (max(hts) - min(hts)) if hts else -1
    if new_spread > 2:
        bad_rows += 1
    print("%4d %5d %6.0f %8.0f %8.0f %8d %8d%s" % (row, len(ss), gh, old_spread,
                                                   old_spread, new_spread, h_spread,
                                                   "  <-- 未对齐" if new_spread > 2 else ""))

print("\n底部极差 >2px 的行数:", bad_rows)
print("RESULT:", "PASS" if bad_rows == 0 else "CHECK")
