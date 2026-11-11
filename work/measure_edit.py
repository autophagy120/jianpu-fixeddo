"""量一次"改音符"到底花在哪：渲染 / PNG 编码 / HTTP 往返。"""
import json
import sys
import time

sys.path.insert(0, r"D:\简谱转调项目")
import numpy as np
import jp.emit_fixed as EF
import jp.library as L
from jp.roi_server import State

it = [x for x in L.load().values() if (x.get("title") or "") == "纯蓝"][0]
t = time.time()
st = State(it["image"], it["ocr_txt"], it.get("key") or "D", title="纯蓝",
           cache_key=it["id"], strength=int(it.get("strength") or 1))
print("State 构建: %.2fs  音符 %d" % (time.time() - t, len(st.pairs)))

bgr1, stats = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc,
                                    mark_missed=True, keys=st.keys_for_pairs())
print("render_fixed_image: %.3fs" % (time.time() - t if False else 0))

for name, fn in (
    ("render_fixed_image 全页", lambda: EF.render_fixed_image(
        st.page, st.pairs, st.key_pc, st.target_pc, mark_missed=True,
        keys=st.keys_for_pairs())),
    ("_render（含 PNG 编码）", lambda: st._render()),
):
    t0 = time.time()
    for _ in range(3):
        r = fn()
    print("%-26s %.3fs/次  (大小 %d KB)" % (name, (time.time() - t0) / 3,
                                            len(r) // 1024 if isinstance(r, bytes) else 0))

# 局部裁剪 + 编码（模拟“只回传改动区域”）
t0 = time.time()
bgr = r if not isinstance(r, bytes) else bgr1
for _ in range(3):
    crop = bgr[1200:1400, 800:1100]
    ok, buf = __import__("cv2").imencode(".png", crop)
print("局部裁剪 + PNG 编码      %.4fs/次  (%d KB)" % ((time.time() - t0) / 3, len(buf) // 1024))
