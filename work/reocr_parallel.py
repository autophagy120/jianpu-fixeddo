"""用 CPU+GPU 并行重跑缺 OCR 的页，写入 library/<id>/ocr.txt。"""
import os
import sys
import time

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.merge_octave as MO
import jp.xiaomi_ocr as XO

MODE = os.environ.get("OCR_MODE", "cpu_gpu")
todo = [i for i, it in L.load().items()
        if not (it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))]
print("方式:", MODE, " 需要 OCR 的页:", len(todo), flush=True)
if not todo:
    sys.exit(0)
items = {i: L.get(i) for i in todo}
paths = [items[i]["image"] for i in todo]
t0 = time.time()
res = XO.ocr_batch(paths, mode=MODE, log=lambda s: print("  " + s, flush=True))
ok = 0
for i in todo:
    raw = res.get(items[i]["image"]) or ""
    txt, cut = MO.trim_degenerate(MO.sanitize_ocr(raw))
    if not txt.strip():
        print("  %s 空结果，跳过" % i, flush=True)
        continue
    p = L.ocr_path_for_id(i)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf8").write(txt)
    L.update(i, ocr_txt=p, status=("done" if items[i].get("out") else "ocr"))
    ok += 1
    print("  %s -> %d 字%s" % (i, len(txt), "（截掉退化 %d 行）" % cut if cut else ""), flush=True)
print("完成 %d/%d 页，用时 %.0fs" % (ok, len(todo), time.time() - t0), flush=True)
