import glob, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jp import replace_img as RI, music as M
from jp import library as LIB

for p in glob.glob(r"out\batch\_ocr\*.txt")[:3]:
    t = io.open(p, encoding="utf8").read()
    pc = RI.KEY_FROM_OCR(t)
    print(p.split("\\")[-1][:36])
    print("   KEY_FROM_OCR ->", pc, M.key_name(pc) if pc is not None else None)
    print("   detect_keys   ->", [k["key"] for k in LIB.detect_keys(t)])

print('parse_key("#F") =', M.parse_key("#F"), ' parse_key("F#") =', M.parse_key("F#"))

try:
    rows = [l for l in io.open(r"out\batch\summary.tsv", encoding="utf8").read().splitlines()
            if "Bad Apple" in l][:3]
    for r in rows:
        print("summary:", r)
except Exception as e:
    print("summary 读取失败:", e)
