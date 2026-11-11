"""量一次渲染到底花在哪：擦除 / 画字 / 找漏识别 / OCR 补字 / PNG 编码。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

t0 = time.time()
from jp import emit_fixed as EF
from jp import core
from jp.roi_server import State

t = time.time()
st = State("BadApple")
print("加载页面（含 OCR/合并）: %.2fs" % (time.time() - t))
print("音符 %d  字形块 %d  页面 %dx%d" % (len(st.pairs), len(st.page.comps),
                                          st.page.gray.shape[1], st.page.gray.shape[0]))

t = time.time()
unc = EF.find_uncovered(st.page, st.pairs)
print("find_uncovered: %.3fs  (%d 个橙框)" % (time.time() - t, len(unc)))

t = time.time()
bgr, stats = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc,
                                   mark_missed=False)
print("render 无 mark_missed: %.3fs" % (time.time() - t))

t = time.time()
bgr2, stats2 = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc,
                                     mark_missed=True)
print("render 含 mark_missed: %.3fs  stats=%s" % (time.time() - t, stats2))

t = time.time()
png = st._render()
print("_render 全流程: %.3fs  (%.1f KB)" % (time.time() - t, len(png) / 1024.0))

t = time.time()
_ = EF.find_uncovered(st.page, st.pairs)
print("再算一次 find_uncovered: %.3fs" % (time.time() - t))
