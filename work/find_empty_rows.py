"""找"整行都没识别出来（只有橙框、没有音符）"的页，用于复现加强识别错行。"""
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.emit_fixed as EF

d = L.load()
alive = [v for v in d.values() if not v.get("deleted")]
rows = []
for it in alive:
    p = f"D:\\简谱转调项目\\library\\{it['id']}\\cache.pkl"
    try:
        c = pickle.load(open(p, "rb"))
        page, pairs = c["page"], c["pairs"]
    except Exception:
        continue
    unc = EF.find_uncovered(page, pairs)
    if len(unc) < 10:
        continue
    # 按 y 归带（一个字形高度左右）
    gh = 30.0
    bands = defaultdict(lambda: [0, 0])
    for s, _o in pairs:
        b = int(((s.box[1] + s.box[3]) / 2) // gh)
        bands[b][0] += 1
    for v in unc:
        b = int(((v.y + v.h / 2)) // gh)
        bands[b][1] += 1
    empty = [(b, n) for b, (np_, n) in bands.items() if np_ == 0 and n >= 3]
    if empty:
        rows.append((len(empty), len(unc), it["id"], it.get("title") or it.get("song") or "", empty[:3]))

rows.sort(reverse=True)
print("有'整行缺失'的页:", len(rows))
for r in rows[:10]:
    print("  块数 %d  橙框 %d  %s %s  %s" % r)
