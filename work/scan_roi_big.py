"""再扫：大框（全宽/半宽/跨两行）会不会一次性删掉大量音符。"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jp import emit_fixed as EF
from jp import ocr_boxes as OB
from jp.roi_server import State

st = State("BadApple")
base = copy.deepcopy(st.pairs)
rows = [r for r in st.page.rows if getattr(r, "kind", "") == "music"]
W = st.page.gray.shape[1]
print("初始音符 %d，音乐行 %d，页宽 %d" % (len(base), len(rows), W))

cases = []
for i, r in enumerate(rows):
    y0, y1 = int(r.y0) - 2, int(r.y1) + 2
    cases.append(("行%d 全宽" % i, (0, y0, W, y1 - y0)))
    cases.append(("行%d 左半" % i, (0, y0, W // 2, y1 - y0)))
    cases.append(("行%d 右半" % i, (W // 2, y0, W - W // 2, y1 - y0)))
    if i + 1 < len(rows):
        y2 = int(rows[i + 1].y1) + 2
        cases.append(("行%d+%d 全宽" % (i, i + 1), (0, y0, W, y2 - y0)))

worst = []
for name, box in cases:
    st.pairs = copy.deepcopy(base)
    before = len(st.pairs)
    st._ocr_roi(*box)
    loss = before - len(st.pairs)
    # 渲染看整行是否被擦白（unpaired_rows 变化）
    bgr, stt = EF.render_fixed_image(st.page, st.pairs, st.key_pc, st.target_pc, mark_missed=False)
    if loss > 0 or stt.get("unpaired_rows", 0) > 9:
        x, y, w, h = box
        crop = st.page.gray[max(0, y):y + h, max(0, x):x + w]
        try:
            boxes, _s = OB.read_boxes(crop)
            chars = OB.chars_from_boxes(boxes, digits_only=False)
            txt = "".join(c.ch for c in chars)
            ndig = sum(1 for c in chars if c.ch.isdigit())
        except Exception as e:
            txt, ndig = "ERR", -1
        worst.append((loss, name, ndig, len(txt), txt[:40], stt.get("unpaired_rows")))

worst.sort(reverse=True)
print("\n%-16s %6s %6s %6s %s" % ("框", "丢音", "数字数", "读到的字", "整行擦白数"))
for loss, name, ndig, ntxt, txt, unr in worst[:14]:
    print("%-16s %6d %6d %6d %-22r %s" % (name, loss, ndig, ntxt, txt, unr))
print("\n会丢音的框数: %d / %d" % (len([w for w in worst if w[0] > 0]), len(cases)))
