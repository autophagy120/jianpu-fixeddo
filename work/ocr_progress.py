"""OCR 重跑进度（读日志型输出用）。"""
import os
import sys
import time

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

todo = [i for i, it in L.load().items()
        if not (it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))]
done = [i for i, it in L.load().items()
        if it.get("ocr_txt") and os.path.exists(it["ocr_txt"])]
print("有文本 %d 页，仍缺 %d 页" % (len(done), len(todo)))
