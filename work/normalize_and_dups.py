"""① 归一化每份的页序（补掉删页留下的空号） ② 找近重复页给建议（不自动删）"""
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

d = L._read_all()
items = d["items"]
gs = d["groups"]
alive = {k: v for k, v in items.items() if not v.get("deleted")}

by_group = defaultdict(list)
for k, v in alive.items():
    by_group[v.get("group") or ""].append(v)

fixed = 0
for gid, pages in by_group.items():
    if not gid:
        continue
    pages.sort(key=lambda p: (p.get("group_pos") or 999, p.get("t0") or 0))
    for idx, p in enumerate(pages, 1):
        if (p.get("group_pos") or 0) != idx:
            p["group_pos"] = idx
            p["group_n"] = len(pages)
            fixed += 1
L._write_all(d)
print("页序归一化：修正 %d 页" % fixed)

# 近重复（dhash 距离 <= 6）
ph = []
for k, v in alive.items():
    h = L._phash_of(v)
    if h:
        ph.append((k, v, h))
print("有指纹的页:", len(ph))

def ham(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")

pairs = []
for i in range(len(ph)):
    for j in range(i + 1, len(ph)):
        if ham(ph[i][2], ph[j][2]) <= 6:
            pairs.append((ph[i], ph[j]))
print("近重复对:", len(pairs))
for (ka, va, _), (kb, vb, _) in pairs[:12]:
    ga = (gs.get(va.get("group") or "") or {}).get("name", "-")
    gb = (gs.get(vb.get("group") or "") or {}).get("name", "-")
    print("  %-8s[%s] ↔ %-8s[%s]  %s | %s"
          % (ka[:8], ga[:10], kb[:8], gb[:10],
             (va.get("origin") or "").split("\\")[-1][:24],
             (vb.get("origin") or "").split("\\")[-1][:24]))
