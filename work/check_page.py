import json
import urllib.request

B = "http://127.0.0.1:8769"
j = json.loads(urllib.request.urlopen(B + "/api/lib", timeout=180).read())
print("库 %d 页 / %d 份 / 未分类 %d / 有标题 %d / 垃圾箱 %d" % (
    len(j["items"]), len(j["groups"]),
    sum(1 for x in j["items"] if not x.get("group")),
    sum(1 for x in j["items"] if x.get("title")), j.get("trash_n")))
h = urllib.request.urlopen(B + "/lib", timeout=30).read().decode("utf8")
checks = {
    "左右两栏": 'id="left"' in h and 'id="right"' in h,
    "拖拽": "grpDrop" in h and "rowDrop" in h,
    "AI 配置": "llmOpen" in h,
    "垃圾箱": "trashOpen" in h,
    "OCR 并行": "CPU+GPU" in h and "最大调度" in h,
    "排序选项": "gsort" in h,
    "缩略图预览": "pvShow" in h,
}
for k, v in checks.items():
    print("  %-10s %s" % (k, "✓" if v else "✗"))
print("AI:", "已配置" if j.get("llm") else "未配置", " OCR:", "可用" if j.get("ocr") else "不可用")
