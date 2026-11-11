"""迁移：把 OCR 文本搬到 library/<id>/ocr.txt；被撞名的页清掉错标题、标回未处理。"""
import json
import os
import shutil
import sys

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L

d = L._read_all()
items = d["items"]
moved = collided = cleared = 0
bad_ids = []
for i, it in items.items():
    newp = L.ocr_path_for_id(i)
    oldp = it.get("ocr_txt") or ""
    if os.path.exists(newp):
        it["ocr_txt"] = newp
        continue
    if oldp and os.path.exists(oldp):
        base = os.path.basename(oldp)
        # 撞名的典型特征：文件名是「歌名__page.txt」（转码后 basename 都叫 page）
        if base.endswith("__page.txt") or base == "page.txt":
            collided += 1
            bad_ids.append(i)
            it["ocr_txt"] = newp          # 指向新位置（暂时不存在 → 需要重新 OCR）
            if it.get("title"):
                it["title"] = ""
                cleared += 1
            if it.get("status") != "done":
                it["status"] = "new"
        else:
            os.makedirs(os.path.dirname(newp), exist_ok=True)
            shutil.copyfile(oldp, newp)
            it["ocr_txt"] = newp
            moved += 1
    else:
        it["ocr_txt"] = newp
L._write_all(d)
print("迁移完成：搬运 %d 页，撞名 %d 页（清掉错标题 %d 条）" % (moved, collided, cleared))
print("撞名的页:", [(items[i].get("song"), items[i].get("image","").split("\\")[-1]) for i in bad_ids][:10])
# 检查是否还有重复的 ocr 路径
import collections
c = collections.Counter(it.get("ocr_txt") for it in items.values())
dup = [(p, n) for p, n in c.items() if n > 1]
print("仍然重复的 OCR 路径:", dup[:5], " 组数:", len(dup))
print("现在没有 OCR 文本的页:", sum(1 for it in items.values()
                                if not (it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))))
