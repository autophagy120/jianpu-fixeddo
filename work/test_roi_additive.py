"""加强识别必须只增不删：连续框两处，前面框里的音一个都不能少。"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jp.roi_server import State

st = State("BadApple")
print("初始音符:", len(st.pairs))


def inside(roi):
    x, y, w, h = roi
    return sum(1 for s, _o in st.pairs
               if s.box[0] >= x - 2 and s.box[2] <= x + w + 2
               and s.box[1] >= y - 2 and s.box[3] <= y + h + 2)


row_notes = sorted([s for s, _o in st.pairs if int(s.row) == 6], key=lambda s: s.box[0])
g1, g2 = row_notes[2:6], row_notes[14:18]


def mk(grp, pad=6):
    return (min(s.box[0] for s in grp) - pad, min(s.box[1] for s in grp) - pad,
            max(s.box[2] for s in grp) - min(s.box[0] for s in grp) + 2 * pad,
            max(s.box[3] for s in grp) - min(s.box[1] for s in grp) + 2 * pad)


roi1, roi2 = mk(g1), mk(g2)
print("框1 %s  内含 %d 音" % ([int(v) for v in roi1], inside(roi1)))
print("框2 %s  内含 %d 音" % ([int(v) for v in roi2], inside(roi2)))

n0 = len(st.pairs)
i1_0 = inside(roi1)
info1 = st.strengthen("ocr", roi1)
n1 = len(st.pairs)
i1_1 = inside(roi1)
print("\n第1次: %s" % info1)
print("  音符 %d -> %d   框1内 %d -> %d" % (n0, n1, i1_0, i1_1))

info2 = st.strengthen("ocr", roi2)
n2 = len(st.pairs)
i1_2, i2_2 = inside(roi1), inside(roi2)
print("\n第2次: %s" % info2)
print("  音符 %d -> %d   框1内 %d   框2内 %d" % (n1, n2, i1_2, i2_2))

# 一行歌词框（OCR 只会读到假名）也不能让音变少
lyr = None
for r in st.page.rows:
    if getattr(r, "kind", "") != "music":
        cs = list(r.comps)
        if len(cs) >= 6:
            lyr = (min(c.x for c in cs) - 4, int(r.y0) - 2,
                   max(c.x + c.w for c in cs) - min(c.x for c in cs) + 8,
                   int(r.y1 - r.y0) + 4)
            break
if lyr:
    nb = len(st.pairs)
    info3 = st.strengthen("ocr", lyr)
    print("\n歌词行框: %s" % info3)
    print("  音符 %d -> %d" % (nb, len(st.pairs)))

ok = (n1 >= n0) and (n2 >= n1) and (i1_2 >= i1_0)
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
