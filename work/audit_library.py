"""体检：看一下现在库里的分组/页序/标题有没有问题。"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

d = L._read_all()
items = d["items"]
gs = d["groups"]
alive = {k: v for k, v in items.items() if not v.get("deleted")}
trash = len(items) - len(alive)

print("=== 总览 ===")
print("页: %d  组: %d  未分类: %d  垃圾箱: %d  墓碑: %d"
      % (len(alive), len(gs), sum(1 for v in alive.values() if not v.get("group")),
         trash, len(d.get("tombstones") or [])))
print("有标题: %d   有 OCR: %d"
      % (sum(1 for v in alive.values() if v.get("title")),
         sum(1 for v in alive.values() if v.get("ocr_txt") and os.path.exists(v["ocr_txt"]))))

by_group = defaultdict(list)
for k, v in alive.items():
    by_group[v.get("group") or ""].append(v)

name_cnt = Counter((g.get("name") or "") for g in gs.values())
print("\n=== 重名组（同一名字多份）===")
dupnames = [(n, c) for n, c in name_cnt.items() if c > 1 and n]
print(dupnames[:10] if dupnames else "无")

print("\n=== 组内页序检查（pos 缺失/重复）===")
bad = []
for gid, g in gs.items():
    pages = by_group.get(gid, [])
    if not pages:
        bad.append((g.get("name"), "空组"))
        continue
    pos = [p.get("group_pos") or 0 for p in pages]
    if sorted(pos) != list(range(1, len(pages) + 1)):
        bad.append((g.get("name"), "页序 %s（%d 页）" % (sorted(pos), len(pages))))
print(bad[:12] if bad else "全部正常")

print("\n=== 组内标题/组名不一致 ===")
mismatch = []
for gid, g in gs.items():
    for p in by_group.get(gid, []):
        t = (p.get("title") or "").strip()
        if t and g.get("name") and t != g.get("name"):
            mismatch.append((g.get("name"), t))
print(mismatch[:10] if mismatch else "无")

print("\n=== 最大的 15 份 ===")
for gid, pages in sorted(by_group.items(), key=lambda kv: -len(kv[1]))[:15]:
    if not gid:
        continue
    g = gs.get(gid) or {}
    pages_s = sorted(pages, key=lambda p: p.get("group_pos") or 0)
    first = (pages_s[0].get("origin") or pages_s[0].get("image") or "").split("\\")[-1]
    print("  %-18s %2d 页  首图=%s" % ((g.get("name") or "?")[:18], len(pages), first[:34]))

print("\n=== 单页的份（%d 个，前 12）===" % sum(1 for gid, p in by_group.items() if gid and len(p) == 1))
n = 0
for gid, pages in by_group.items():
    if not gid or len(pages) != 1:
        continue
    g = gs.get(gid) or {}
    print("  %-20s 图=%s" % ((g.get("name") or "?")[:20],
                            (pages[0].get("origin") or "").split("\\")[-1][:40]))
    n += 1
    if n >= 12:
        break

print("\n=== 未分类的页（%d）===" % sum(1 for v in alive.values() if not v.get("group")))
for v in [x for x in alive.values() if not x.get("group")][:15]:
    print("  %-8s song=%-10s title=%-10s 图=%s"
          % (v["id"][:8], (v.get("song") or "")[:10], (v.get("title") or "-")[:10],
             (v.get("origin") or "").split("\\")[-1][:40]))

print("\n=== 最近手动改动过的 8 页（按 updated）===")
rec = sorted(alive.values(), key=lambda v: v.get("updated") or "", reverse=True)[:8]
for v in rec:
    print("  %s  %-10s 组=%-12s status=%s"
          % (v.get("updated") or "-", (v.get("title") or v.get("song") or "")[:10],
             (gs.get(v.get("group") or "") or {}).get("name", "-")[:12], v.get("status")))
