"""实测 gpu_first 调度：3 张小图，看 GPU 是否抢在 CPU 前面出结果。"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.xiaomi_ocr as X

# 造 3 张"中等大小"的图（裁自库里的谱）
src = [v["image"] for v in L.load().values() if not v.get("deleted") and v.get("image")][0]
bgr = cv2.imdecode(np.frombuffer(open(src, "rb").read(), np.uint8), cv2.IMREAD_COLOR)
H, W = bgr.shape[:2]
tmpdir = os.path.join(r"D:\简谱转调项目\out", "_ocr_sched")
os.makedirs(tmpdir, exist_ok=True)
paths = []
for k in range(3):
    y0 = min(H - 1, int(H * k / 3))
    sub = bgr[y0:y0 + H // 3, :]
    p = os.path.join(tmpdir, "s%d.png" % k)
    ok, buf = cv2.imencode(".png", sub)
    buf.tofile(p)
    paths.append(p)
print("任务:", len(paths), "张  尺寸:", sub.shape)

t0 = time.time()
res = X.ocr_batch(paths, mode="gpu_first", max_tokens=600,
                  log=lambda s: print("  " + s, flush=True))
dt = time.time() - t0
ok = sum(1 for v in res.values() if v and len(v) > 20)
print("完成 %d/%d，用时 %.0fs（%.0fs/页）" % (ok, len(paths), dt, dt / len(paths)))
