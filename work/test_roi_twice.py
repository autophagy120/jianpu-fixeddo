"""复现：连续两次"加强识别"后，第一个框里的音符为什么也消失。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jp import emit_fixed as EF
from jp.roi_server import State

st = State("BadApple")
print("初始音符:", len(st.pairs))


def inside(roi):
    x, y, w, h = roi
    n = 0
    for s, _o in st.pairs:
        if (s.box[0] >= x - 2 and s.box[2] <= x + w + 2
                and s.box[1] >= y - 2 and s.box[3] <= y + h + 2):
            n += 1
    return n


# 同一行取两段，模拟用户先后框两处
row_notes = sorted([s for s, _o in st.pairs if int(s.row) == 6], key=lambda s: s.box[0])
g1 = row_notes[2:6]
g2 = row_notes[14:18]
roi1 = (min(s.box[0] for s in g1) - 6, min(s.box[1] for s in g1) - 8,
        max(s.box[2] for s in g1) - min(s.box[0] for s in g1) + 12,
        max(s.box[3] for s in g1) - min(s.box[1] for s in g1) + 16)
roi2 = (min(s.box[0] for s in g2) - 6, min(s.box[1] for s in g2) - 8,
        max(s.box[2] for s in g2) - min(s.box[0] for s in g2) + 12,
        max(s.box[3] for s in g2) - min(s.box[1] for s in g2) + 16)
print("框1 %s  框2 %s" % ([int(v) for v in roi1], [int(v) for v in roi2]))
print("框1内原有音符 %d 个，框2内 %d 个" % (inside(roi1), inside(roi2)))

st.strengthen("ocr", roi1)
print("\n[第1次加强识别后]")
print("  总音符 %d   框1内 %d   框2内 %d" % (len(st.pairs), inside(roi1), inside(roi2)))
b1, stt = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc, mark_missed=True)
print("  渲染 stats:", {k: stt.get(k) for k in ("drawn", "unpaired_rows", "uncovered")})

st.strengthen("ocr", roi2)
print("\n[第2次加强识别后]")
print("  总音符 %d   框1内 %d   框2内 %d" % (len(st.pairs), inside(roi1), inside(roi2)))
b2, stt2 = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc, mark_missed=True)
print("  渲染 stats:", {k: stt2.get(k) for k in ("drawn", "unpaired_rows", "uncovered")})
print("  被整行擦白的行数 1→2 次:", stt.get("unpaired_rows"), "->", stt2.get("unpaired_rows"))

# 行号分布：ROI 补出来的音 row 是什么
from collections import Counter
print("\n音符 row 分布:", Counter(int(s.row) for s, _o in st.pairs).most_common(6))
