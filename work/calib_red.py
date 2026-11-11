"""标定：怎么把"已经转调过的输出图"从原始谱面里认出来（避免重复处理）。"""
import os
import numpy as np
import cv2

ROOT = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱"


def load(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def metrics(p):
    b = load(p)
    if b is None:
        return None
    B, G, R = [b[:, :, i].astype(int) for i in range(3)]
    red = (R > 100) & (R - G > 60) & (R - B > 60)
    strong = (R > 140) & (G < 70) & (B < 70)
    ink = (R + G + B) < 420
    ni = max(1, int(ink.sum()))
    return dict(red=float(red.sum()) / ni, strong=float(strong.sum()) / ni,
                nred=int(red.sum()), nstr=int(strong.sum()))


rows = []
for d in sorted(os.listdir(ROOT))[:20]:
    dp = os.path.join(ROOT, d)
    if not os.path.isdir(dp):
        continue
    for f in sorted(os.listdir(dp)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        r = metrics(os.path.join(dp, f))
        if r:
            rows.append((("转调" if "(原调" in f else "原始"), f[:32],
                         r["red"], r["strong"], r["nred"], r["nstr"]))

orig = [r for r in rows if r[0] == "原始"]
out = [r for r in rows if r[0] == "转调"]
print("原始页 %d 张，转调输出 %d 张" % (len(orig), len(out)))

def show(rs, rev, n=5):
    for t, f, a, b, c, d in sorted(rs, key=lambda x: x[2], reverse=rev)[:n]:
        print("  %-4s red=%.4f strong=%.4f  nred=%6d nstr=%6d  %s" % (t, a, b, c, d, f))

print("--- 原始页里 red 比最高的 ---")
show(orig, True)
print("--- 转调输出里 red 比最低的 ---")
show(out, False)
print()
print("原始 strong 比 max = %.4f" % max(x[3] for x in orig))
print("转调 strong 比 min = %.4f" % min(x[3] for x in out))
print("原始 red 比    max = %.4f" % max(x[2] for x in orig))
print("转调 red 比    min = %.4f" % min(x[2] for x in out))

# 判据：纯红像素足够多，且占比够高
def is_transposed(m):
    return m["strong"] > 0.06 and m["nstr"] > 3000

fp = [x[1] for x in orig if is_transposed(dict(red=x[2], strong=x[3], nred=x[4], nstr=x[5]))]
fn = [x[1] for x in out if not is_transposed(dict(red=x[2], strong=x[3], nred=x[4], nstr=x[5]))]
print("\n判据 strong>0.06 且 nstr>3000")
print("  原始页误判:", len(fp), fp[:3])
print("  转调页漏判:", len(fn), fn[:3])
