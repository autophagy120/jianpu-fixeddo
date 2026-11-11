"""修复：把"已 OCR 但谱行为空"的页，用 RapidOCR 数字行重建文本。

跑完会统计：处理了多少页、其中多少页现在有 >=2 行谱行。
"""
import os
import sys
import time

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

todo = []
for i, it in L.load().items():
    if it.get("deleted"):
        continue
    tp = it.get("ocr_txt") or ""
    if not tp or not os.path.exists(tp):
        continue
    txt = open(tp, encoding="utf8").read()
    if L.music_line_count(txt) < 2:
        todo.append((i, it, txt))
print("谱行不足的页:", len(todo), flush=True)

fixed = 0
t0 = time.time()
for n, (i, it, txt) in enumerate(todo, 1):
    try:
        new = L.ensure_music_text(it, txt, log=lambda m: None)
        if L.music_line_count(new) >= 2:
            tp = L.ocr_path_for_id(i)
            os.makedirs(os.path.dirname(tp), exist_ok=True)
            open(tp, "w", encoding="utf8").write(new)
            L.update(i, ocr_txt=tp, status=("done" if it.get("out") else "ocr"))
            fixed += 1
            print("[%d/%d] %s 修复（%d 行谱行）" % (n, len(todo), i[:8], L.music_line_count(new)), flush=True)
        else:
            print("[%d/%d] %s 仍无谱行（可能不是简谱）" % (n, len(todo), i[:8]), flush=True)
    except Exception as e:
        print("[%d/%d] %s 失败：%s" % (n, len(todo), i[:8], str(e)[:100]), flush=True)
print("完成：修复 %d / %d 页，用时 %.0fs" % (fixed, len(todo), time.time() - t0), flush=True)
