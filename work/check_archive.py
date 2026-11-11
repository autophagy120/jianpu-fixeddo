import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jp.library as L

items = L.load()
lib = os.path.abspath(L.LIB_DIR).lower()
inside = [x for x in items.values()
          if os.path.abspath(x.get("image", "")).lower().startswith(lib)]
outside = [x for x in items.values()
           if not os.path.abspath(x.get("image", "")).lower().startswith(lib)]
print("总条目 %d，已复制进库 %d，仍引用外部 %d" % (len(items), len(inside), len(outside)))
if outside:
    for x in outside[:3]:
        print("  外部:", x["song"], x["image"][:80], "存在:", os.path.exists(x["image"]))

dirs = [d for d in os.listdir(L.LIB_DIR) if os.path.isdir(os.path.join(L.LIB_DIR, d))] \
    if os.path.isdir(L.LIB_DIR) else []
tot = 0
for d in dirs:
    for f in os.listdir(os.path.join(L.LIB_DIR, d)):
        tot += os.path.getsize(os.path.join(L.LIB_DIR, d, f))
print("library 目录：%d 个条目目录，%.1f MB" % (len(dirs), tot / 1048576.0))
caches = [d for d in dirs if os.path.exists(L.cache_path(d))]
print("已有打开缓存的页: %d" % len(caches))
