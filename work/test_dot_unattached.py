"""进程内验证：原谱有点、但几何没挂到音上（dot_boxes 为空）→ 标成高/低音后黑点应被擦掉。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\简谱转调项目")
import jp.emit_fixed as EF
import jp.library as L
from jp.roi_server import State

pid = "37d477b51f0f"
it = L.get(pid)
st = State(it["image"], it.get("ocr_txt"), it.get("key") or "D", cache_key=pid)
print("音符", len(st.pairs), " med_h", st.med_h)
gh = st.med_h

# 找"点状连通块 + 附近 oct=0 的音 + 该点不在这个音的 dot_boxes 里"
dots = [c for c in st.page.comps
        if 0.10 * gh <= c.h <= 0.55 * gh and c.w <= 0.75 * c.h]
print("点状连通块:", len(dots))
found = None
for c in dots:
    cx, cy = c.x + c.w / 2, c.y + c.h / 2
    for s, o in st.pairs:
        if int(o.octave) != 0 or str(o.digit) == "0":
            continue
        px = (s.box[0] + s.box[2]) / 2
        py = (s.box[1] + s.box[3]) / 2
        if abs(cx - px) > 0.5 * gh or not (0.5 * gh < abs(cy - py) < 1.7 * gh):
            continue
        if any(b[0] - 2 <= c.x and c.x + c.w <= b[2] + 2 and b[1] - 2 <= c.y and c.y + c.h <= b[3] + 2
               for b in s.dot_boxes):
            continue
        found = (s, o, c)
        break
    if found:
        break

if not found:
    print("没找到'有点但没挂到音上'的样本")
    raise SystemExit(0)

s, o, c = found
print("样本: digit=%s oct=%d dot_box=(%d,%d,%d,%d)"
      % (o.digit, o.octave, c.x, c.y, c.w, c.h))

def dark_in(bgr, box, pad=2):
    x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    x1, y1 = box[0] + box[2] + pad, box[1] + box[3] + pad
    sub = bgr[y0:y1, x0:x1]
    return int((sub.sum(axis=2) < 300).sum())

b0, _ = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc, mark_missed=True)
print("标之前该点区域黑像素:", dark_in(b0, (c.x, c.y, c.w, c.h)))

idx = next(i for i, (s2, _o) in enumerate(st.pairs) if s2 is s)
st.set_oct(idx, 1 if (s.box[1] + s.box[3]) / 2 > c.y else -1)
b1, _ = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc, mark_missed=True)
print("标之后该点区域黑像素:", dark_in(b1, (c.x, c.y, c.w, c.h)))
print("黑点已擦:", dark_in(b1, (c.x, c.y, c.w, c.h)) < dark_in(b0, (c.x, c.y, c.w, c.h)))
