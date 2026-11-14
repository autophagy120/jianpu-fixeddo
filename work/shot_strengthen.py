"""看图核对：加强识别前后，ROI 区域的渲染对比（裁图放大）。"""
import sys

import cv2

sys.path.insert(0, r"D:\简谱转调项目")
import jp.emit_fixed as EF
import jp.library as L
from jp.roi_server import State

pid = "0983f840012c"
it = L.get(pid)
st = State(it["image"], it.get("ocr_txt"), it.get("key") or "C", cache_key=pid)
unc = st.uncovered
print("音符", len(st.pairs), "橙框", len(unc))

roi = None
added = []
for u in unc[:15]:
    cy = u.y + u.h / 2
    same = [v for v in unc if abs((v.y + v.h / 2) - cy) < 30 and abs(v.x - u.x) < 600]
    if len(same) < 3:
        continue
    x0 = max(0, min(v.x for v in same) - 15)
    x1 = min(st.page.gray.shape[1], max(v.x + v.w for v in same) + 15)
    box = [int(x0), int(cy - 45), int(x1 - x0), 90]
    before = {(str(o.digit), round(s.box[0]), round(s.box[1])) for s, o in st.pairs}
    st._ocr_roi(box[0], box[1], box[2], box[3])
    added = [(s, o) for s, o in st.pairs
             if (str(o.digit), round(s.box[0]), round(s.box[1])) not in before]
    if added:
        roi = box
        break

print("ROI:", roi, " 新增:", [(o.digit, round(s.box[0]), round(s.box[1]), int(s.row)) for s, o in added])
if not roi:
    raise SystemExit(0)

bgr, _ = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc, mark_missed=True)
x, y, w, h = roi
pad = 30
crop = bgr[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4)
out = r"D:\简谱转调项目\out\_strengthen_after.png"
cv2.imencode(".png", crop)[1].tofile(out)
print("裁图:", out, crop.shape)
