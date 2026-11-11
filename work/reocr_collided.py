"""给撞名的那 24 页重新跑本地 OCR（各自写进 library/<id>/ocr.txt）。"""
import os
import sys
import time

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.xiaomi_ocr as XO

print("Xiaomi-OCR 可用:", XO.available(), flush=True)
if not XO.available():
    sys.exit("OCR 不可用，先启动本地模型")

todo = [i for i, it in L.load().items()
        if not (it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))]
print("需要重新 OCR 的页:", len(todo), flush=True)
ok = fail = 0
t0 = time.time()
for n, i in enumerate(todo, 1):
    it = L.get(i)
    if not it:
        continue
    try:
        t = time.time()
        txt = L.ensure_ocr(it)
        L.update(i, status=("done" if it.get("out") else "ocr"))
        ok += 1
        print("[%d/%d] %s  %d 字  %.0fs" % (n, len(todo), i, len(txt), time.time() - t), flush=True)
    except Exception as e:
        fail += 1
        print("[%d/%d] %s 失败: %s" % (n, len(todo), i, str(e)[:120]), flush=True)
print("完成 ok=%d fail=%d 用时 %.0fs" % (ok, fail, time.time() - t0), flush=True)

got = sum(1 for it in L.load().values()
          if it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))
print("现在有 OCR 文本的页:", got, flush=True)
