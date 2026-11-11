import os
import re
import shutil

ROOT = r"D:\简谱转调项目"
JP = os.path.join(ROOT, "jp")
DEST = os.path.join(ROOT, "experiments")
files = sorted(f for f in os.listdir(JP) if f.endswith(".py"))
src = {f: open(os.path.join(JP, f), encoding="utf8").read() for f in files}

used = set()
for f, s in src.items():
    for m in re.finditer(r"(?:jp\.|from\s+\.\s*|from\s+\.)([a-z_][a-z0-9_]*)", s):
        used.add(m.group(1))
    for m in re.finditer(r"from\s+\.\s+import\s+([\w,\s]+)", s):
        for t in m.group(1).split(","):
            used.add(t.strip())

keep_always = {"roi_server", "library", "core", "align", "music", "merge_octave",
               "emit_fixed", "glyphnet", "ocr_boxes", "xiaomi_ocr", "llm", "edits",
               "batch", "train_glyph", "__init__", "measures", "text_out", "settings",
               "utils", "replace_img", "pretrained", "synth"}

move = []
for f in files:
    name = os.path.splitext(f)[0]
    if name in keep_always or name in used:
        continue
    move.append(f)

os.makedirs(DEST, exist_ok=True)
for f in move:
    shutil.move(os.path.join(JP, f), os.path.join(DEST, f))
print("移到 experiments/：%d 个" % len(move))
print("  " + " ".join(os.path.splitext(f)[0] for f in move))
left = sorted(os.path.splitext(f)[0] for f in os.listdir(JP) if f.endswith(".py"))
print("\njp/ 保留 %d 个：" % len(left))
print("  " + " ".join(left))
