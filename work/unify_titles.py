"""把这 5 页的标题统一成份名（界面上显示的是页标题）。"""
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

d = L._read_all()
gs = d["groups"]
alive = {k: v for k, v in d["items"].items() if not v.get("deleted")}

fixed = []
for k, v in alive.items():
    gid = v.get("group") or ""
    g = gs.get(gid) or {}
    gname = (g.get("name") or "").strip()
    t = (v.get("title") or "").strip()
    if gname and t and t != gname:
        v["title"] = gname
        v["updated"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
        fixed.append((k, t, gname))

L._write_all(d)
print("统一标题:", len(fixed), "页")
for k, old, new in fixed:
    print("   %s  %s → %s" % (k[:8], old, new))

# 再检查一遍
d2 = L._read_all()
gs2 = d2["groups"]
left = [(k, (v.get("title") or ""), (gs2.get(v.get("group") or "") or {}).get("name"))
        for k, v in d2["items"].items()
        if not v.get("deleted") and v.get("group") and (v.get("title") or "")
        and v.get("title") != (gs2.get(v.get("group") or "") or {}).get("name")]
print("剩余不一致:", left if left else "0")
