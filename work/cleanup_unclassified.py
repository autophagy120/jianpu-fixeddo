"""清理：把"同一张图的不同格式副本"（未分类那一份）移进垃圾箱；孤立页归组。"""
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

d = L._read_all()
items = d["items"]
gs = d["groups"]
alive = {k: v for k, v in items.items() if not v.get("deleted")}

# 同源同图（文件名去掉扩展名后一致）→ 未分类那份删掉
def base_name(v):
    o = (v.get("origin") or v.get("image") or "").split("\\")[-1]
    for suf in (" [原始大小]", "（原调", "_Page"):
        o = o.split(suf)[0]
    return o.rsplit(".", 1)[0].strip().lower()

by_name = defaultdict(list)
for k, v in alive.items():
    by_name[base_name(v)].append(v)

removed = []
for bn, vs in by_name.items():
    if len(vs) < 2:
        continue
    free = [v for v in vs if not v.get("group")]
    grouped = [v for v in vs if v.get("group")]
    if free and grouped:
        for v in free:
            L.soft_delete([v["id"]])
            removed.append((v["id"], bn, (gs.get(grouped[0].get("group") or "") or {}).get("name", "")))

print("删掉的同图副本:", len(removed))
for i, bn, gname in removed:
    print("   %s  (%s)  已并入的份: %s" % (i[:8], bn[:34], gname))

# 剩下仍未分类的页：看有没有同名组，有就并进去，没有就建一份
d = L._read_all()
alive = {k: v for k, v in d["items"].items() if not v.get("deleted")}
free = [v for v in alive.values() if not v.get("group")]
print("\n仍未分类:", len(free))
made = 0
for v in free:
    key = (v.get("title") or v.get("song") or "").strip()
    if not key:
        print("   %s 没有标题，留在右侧" % v["id"][:8])
        continue
    hit = [(gid, g) for gid, g in d["groups"].items()
           if (g.get("name") or "").strip() == key]
    if hit:
        gid = hit[0][0]
        L.move_to([v["id"]], gid)
        print("   %s → 并入已有份「%s」" % (v["id"][:8], key))
    else:
        g = L.group_new(key)
        L.move_to([v["id"]], g["id"])
        print("   %s → 新建份「%s」" % (v["id"][:8], key))
    made += 1

d = L._read_all()
alive = {k: v for k, v in d["items"].items() if not v.get("deleted")}
print("\n最终：页 %d  组 %d  未分类 %d  垃圾箱 %d"
      % (len(alive), len(d["groups"]),
         sum(1 for v in alive.values() if not v.get("group")),
         sum(1 for v in d["items"].values() if v.get("deleted"))))
