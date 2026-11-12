"""统计出图质量：每页音符数 / 输出文件大小，挑出最值得人工看的页。"""
import os
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

d = L.load()
alive = [v for v in d.values() if not v.get("deleted")]
rows = []
for v in alive:
    out = v.get("out") or ""
    size = os.path.getsize(out) // 1024 if out and os.path.exists(out) else 0
    rows.append(dict(id=v["id"], title=(v.get("title") or v.get("song") or "")[:16],
                     notes=int(v.get("notes") or 0), size=size,
                     status=v.get("status"), strength=v.get("strength"),
                     out=out))

rows.sort(key=lambda r: r["notes"])
print("存活 %d 页；有输出 %d 页" % (len(rows), sum(1 for r in rows if r["size"])))
print("\n=== 音符最少的 15 页（最该先看）===")
for r in rows[:15]:
    print("  %-8s %-16s 音符 %4d  强度 %-4s  %s"
          % (r["id"][:8], r["title"], r["notes"], r["strength"],
             (os.path.basename(r["out"]) if r["out"] else "(无输出)")[:34]))

import statistics
ns = [r["notes"] for r in rows if r["notes"] > 0]
print("\n音符数：中位 %d，平均 %.0f，最多 %d，为 0 的页 %d"
      % (statistics.median(ns) if ns else 0, statistics.mean(ns) if ns else 0,
         max(ns) if ns else 0, sum(1 for r in rows if r["notes"] == 0)))
print("输出总大小 %.1f MB" % (sum(r["size"] for r in rows) / 1024.0))
