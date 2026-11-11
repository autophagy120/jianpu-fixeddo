"""扫描：哪些框会让"加强识别"把原有的音删掉却补不回来（先删后补的漏洞）。"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jp import emit_fixed as EF
from jp import ocr_boxes as OB
from jp.roi_server import State

st = State("BadApple")
base = copy.deepcopy(st.pairs)
print("初始音符:", len(base), " 页面行:", len(st.page.rows))

rows = [r for r in st.page.rows if getattr(r, "kind", "") == "music"]
print("音乐行:", len(rows))

worst = []
for ri, r in enumerate(rows):
    xs = [c.x for c in r.comps]
    if not xs:
        continue
    x_lo, x_hi = min(xs), max(xs) + 1
    span = x_hi - x_lo
    for k in range(3):
        x0 = int(x_lo + span * k / 3.0)
        x1 = int(x_lo + span * (k + 1) / 3.0)
        box = (x0, int(r.y0) - 2, x1 - x0, int(r.y1 - r.y0) + 4)
        st.pairs = copy.deepcopy(base)
        before = len(st.pairs)
        try:
            st._ocr_roi(*box)
        except Exception as e:
            print("  异常", box, e)
            continue
        after = len(st.pairs)
        if before - after > 0:
            worst.append((before - after, box, ri, k, after))

worst.sort(reverse=True)
print("\n会丢音符的框（丢得最多的前 10）：")
for loss, box, ri, k, after in worst[:10]:
    x, y, w, h = box
    # 这一框 OCR 到底读到了什么
    crop = st.page.gray[max(0, y):y + h, max(0, x):x + w]
    try:
        boxes, _s = OB.read_boxes(crop)
        chars = OB.chars_from_boxes(boxes, digits_only=False)
        txt = "".join(c.ch for c in chars)
        ndig = sum(1 for c in chars if c.ch.isdigit())
    except Exception as e:
        txt, ndig = "ERR:%s" % e, -1
    print("  丢 %2d 个  行%d 第%d段 box=(%d,%d,%d,%d)  OCR读到 %r  数字%d  剩%d"
          % (loss, ri, k, x, y, w, h, txt[:24], ndig, after))

print("\n共 %d 个框会丢音符" % len(worst))
