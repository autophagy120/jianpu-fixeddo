import os
import re

ROOT = r"D:\简谱转调项目"
JP = os.path.join(ROOT, "jp")
files = [f for f in os.listdir(JP) if f.endswith(".py")]

pat = re.compile(r"(?:from\s+jp\.(\w+)|from\s+\.(\w+)|import\s+jp\.(\w+)|from\s+\.\s+import\s+([\w, ]+))")


def deps(name):
    p = os.path.join(JP, name + ".py")
    if not os.path.exists(p):
        return set()
    src = open(p, encoding="utf8").read()
    out = set()
    for m in pat.finditer(src):
        g = [x for x in m.groups() if x]
        for x in g:
            if "," in x:
                out |= {t.strip() for t in x.split(",") if t.strip()}
            else:
                out.add(x.strip())
    return out


seeds = ["roi_server", "library", "batch", "train_glyph", "edits"]
seen, todo = set(), list(seeds)
while todo:
    n = todo.pop()
    if n in seen:
        continue
    seen.add(n)
    for d in deps(n):
        if d + ".py" in files and d not in seen:
            todo.append(d)

keep = sorted(seen)
rest = sorted(os.path.splitext(f)[0] for f in files if os.path.splitext(f)[0] not in seen)
print("保留 %d 个:" % len(keep))
print("  " + " ".join(keep))
print("\n移到 experiments/ 的 %d 个:" % len(rest))
print("  " + " ".join(rest))
open(os.path.join(ROOT, "work", "_keep.txt"), "w", encoding="utf8").write("\n".join(keep))
open(os.path.join(ROOT, "work", "_rest.txt"), "w", encoding="utf8").write("\n".join(rest))
