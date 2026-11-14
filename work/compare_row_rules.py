"""对比：旧 _row_for（最近音）vs 新规则（行范围/合成行）对橙框位置的行基线误差。"""
import pickle
import statistics
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.emit_fixed as EF

d = L.load()
ids = [k for k, v in d.items() if not v.get("deleted") and v.get("ocr_txt")][:25]
tot = worse_old = 0
for pid in ids:
    try:
        c = pickle.load(open(f"D:\\简谱转调项目\\library\\{pid}\\cache.pkl", "rb"))
        page, pairs = c["page"], c["pairs"]
    except Exception:
        continue
    unc = EF.find_uncovered(page, pairs)
    if not unc:
        continue
    gh = statistics.median([max(1.0, s.box[3] - s.box[1]) for s, _ in pairs]) if pairs else 30.0
    # 旧规则：最近音的 row
    row_base = {}
    for s, _o in pairs:
        row_base.setdefault(int(s.row), []).append(float(s.box[3]))
    row_base = {r: statistics.median(v) for r, v in row_base.items()}
    # 新规则：行范围 → 否则合成（基线≈该框自身底边）
    rows = {}
    for s, _o in pairs:
        rows.setdefault(int(s.row), []).append(s.box)
    for v in unc:
        cy = (v.y + v.h / 2)
        # old
        best, bd = -1, 1e18
        for s, _o in pairs:
            dy = abs((s.box[1] + s.box[3]) / 2 - cy)
            if dy < bd:
                bd, best = dy, int(s.row)
        old_err = abs(row_base.get(best, cy) - cy) if best >= 0 else 0
        # new
        nb = None
        for r, boxes in rows.items():
            y0 = min(b[1] for b in boxes) - 0.5 * gh
            y1 = max(b[3] for b in boxes) + 0.5 * gh
            if y0 <= cy <= y1:
                nb = r
                break
        if nb is None:
            for s, _o in pairs:
                dy = abs((s.box[1] + s.box[3]) / 2 - cy)
                if dy <= 0.6 * gh:
                    nb = int(s.row)
                    break
        new_err = abs(row_base.get(nb, cy) - cy) if nb is not None else 0
        tot += 1
        if old_err > 40 and new_err <= 40:
            worse_old += 1
print("橙框采样 %d，旧规则会画偏 >40px 而新规则不偏的：%d" % (tot, worse_old))
