"""用最大调度（3 GPU + 3 CPU）补齐缺 OCR 的页，边跑边落盘。"""
import os
import sys
import time

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.xiaomi_ocr as XO

MODE = os.environ.get("OCR_MODE", "max")
todo = [i for i, it in L.load().items()
        if not (it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))]
print("方式:", MODE, XO.plan_workers(MODE), " 缺 OCR 的页:", len(todo), flush=True)
if not todo:
    sys.exit(0)
t0 = time.time()
res = L.ocr_batch_ids(todo, mode=MODE, log=lambda s: print("  " + str(s), flush=True))
print("完成:", res, " 用时 %.0fs" % (time.time() - t0), flush=True)

got = sum(1 for it in L.load().values()
          if it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))
print("现在有 OCR 文本:", got, "页", flush=True)
