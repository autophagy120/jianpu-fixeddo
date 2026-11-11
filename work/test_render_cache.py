"""验证提速改造没有改变输出：向量化擦除 == 老的纯 Python 擦除；缓存命中 == 缓存未命中。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from jp import emit_fixed as EF
from jp.roi_server import State


def bg_reference(page, pairs, med_h, comps):
    """改造前的原版实现（纯 Python 双重循环），作为等价性基准。"""
    gray = page.gray.copy()
    for slot, o in pairs:
        bx = slot.box
        ex = 0.45 * med_h
        ebx = (bx[0] - ex, bx[1] - ex, bx[2] + ex, bx[3] + ex)
        for c in comps:
            if c.h < 0.18 * med_h or c.w > 2.5 * c.h:
                continue
            inside = (c.x >= ebx[0] and c.x + c.w <= ebx[2]
                      and c.y >= ebx[1] and c.y + c.h <= ebx[3])
            ox = min(c.x + c.w, bx[2]) - max(c.x, bx[0])
            oy = min(c.y + c.h, bx[3]) - max(c.y, bx[1])
            overlap = ox > 0 and oy > 0 and ox * oy >= 0.20 * (c.w * c.h)
            if inside or overlap:
                EF._erase_boxes(gray, [c.box], pad=1)
        extra = list(slot.dot_boxes)
        if slot.acc_box is not None:
            extra.append(slot.acc_box)
        if extra:
            EF._erase_boxes(gray, extra, pad=1)
    EF.erase_unpaired_rows(page, pairs, gray)
    return gray


t = time.time()
st = State("BadApple")
print("加载页面: %.1fs   音符 %d  连通块 %d" % (time.time() - t, len(st.pairs), len(st.page.comps)))

page, pairs = st.page, st.pairs
med_h = float(np.median([max(1.0, s.box[3] - s.box[1]) for s, _o in pairs]))
comps = page.comps

t = time.time()
ref = bg_reference(page, pairs, med_h, comps)
t_ref = time.time() - t
print("参考实现（旧算法）: %.3fs" % t_ref)

EF._BG_CACHE.clear()
t = time.time()
new, _bands = EF._build_background(page, pairs, "comp", med_h, comps)
t_new = time.time() - t
print("新实现（首次，含向量化）: %.3fs" % t_new)
print("底图与旧算法逐像素相同:", bool(np.array_equal(ref, new)))
print("加速: %.1fx" % (t_ref / t_new if t_new else 0))

# 缓存命中：数字/字号改动不改变几何 → 应当直接复用
EF._BG_CACHE.clear()
t = time.time()
bgr1, _ = EF.render_fixed_image(page, pairs, st.key_pc, st.target_pc, mark_missed=True)
t1 = time.time() - t
t = time.time()
bgr2, _ = EF.render_fixed_image(page, pairs, st.key_pc, st.target_pc, mark_missed=True)
t2 = time.time() - t
print("首次渲染（要擦底图）: %.3fs" % t1)
print("第二次渲染（底图命中缓存）: %.3fs   加速 %.1fx" % (t2, t1 / t2 if t2 else 0))
print("两次渲染结果逐像素相同:", bool(np.array_equal(bgr1, bgr2)))

# 只改字号（几何不变）→ 走缓存，且与"强制重算"结果一致
s0 = pairs[10][0]
old_cls = getattr(s0, "size_class", "main")
s0.size_class = "aux"
t = time.time()
bgr_aux, _ = EF.render_fixed_image(page, pairs, st.key_pc, st.target_pc, mark_missed=True)
t3 = time.time() - t
EF._BG_CACHE.clear()
bgr_aux2, _ = EF.render_fixed_image(page, pairs, st.key_pc, st.target_pc, mark_missed=True)
s0.size_class = old_cls
print("改字号后渲染: %.3fs（命中缓存）  与强制重算一致: %s" % (t3, bool(np.array_equal(bgr_aux, bgr_aux2))))

t = time.time()
png = st._render()
print("_render 全流程（缓存热）: %.3fs  %.0f KB" % (time.time() - t, len(png) / 1024.0))

ok = np.array_equal(ref, new) and np.array_equal(bgr1, bgr2) and np.array_equal(bgr_aux, bgr_aux2)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
