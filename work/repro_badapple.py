"""在 Bad Apple!!（有整行缺失）上复现加强识别后的行归属。"""
import pickle
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.emit_fixed as EF
from jp.roi_server import State

pid = "985bd1187a86"
it = L.get(pid)
st = State(it["image"], it.get("ocr_txt"), it.get("key") or "C", cache_key=pid)
print("音符", len(st.pairs), "橙框", len(st.uncovered), "med_h", st.med_h, "图高", st.page.gray.shape[0])

# 用 find_uncovered 的结果找到"整行只有橙框"的簇
unc = st.uncovered
gh = 30.0
bands = {}
for v in unc:
    b = int(((v.y + v.h / 2)) // gh)
    bands.setdefault(b, []).append(v)
cand = None
for b, vs in bands.items():
    cy = sum(v.y + v.h / 2 for v in vs) / len(vs)
    near = [s for s, _ in st.pairs if abs((s.box[1] + s.box[3]) / 2 - cy) < 0.9 * st.med_h]
    if len(vs) >= 3 and not near:
        cand = (b, cy, vs)
        break
print("整行缺失带:", cand[0] if cand else None, " 橙框", len(cand[2]) if cand else 0)

if cand:
    b, cy, vs = cand
    x0 = max(0, min(v.x for v in vs) - 20)
    x1 = min(st.page.gray.shape[1], max(v.x + v.w for v in vs) + 20)
    roi = [int(x0), int(cy - 50), int(x1 - x0), 100]
    print("ROI:", roi)
    before = {(str(o.digit), round(s.box[0]), round(s.box[1])) for s, o in st.pairs}
    info = st._ocr_roi(roi[0], roi[1], roi[2], roi[3])
    print("_ocr_roi:", info)
    added = [(s, o) for s, o in st.pairs
             if (str(o.digit), round(s.box[0]), round(s.box[1])) not in before]
    print("真正新增", len(added))
    for s, o in added[:10]:
        c = (s.box[1] + s.box[3]) / 2
        print("   row=%-6s box_y=%.0f center_y=%.0f (相对该行 %+.0f) digit=%s"
              % (s.row, s.box[1], c, c - cy, o.digit))
