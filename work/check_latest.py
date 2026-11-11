import json
import urllib.request

B = "http://127.0.0.1:8769"
lib = urllib.request.urlopen(B + "/lib", timeout=60).read().decode("utf8")
roi = urllib.request.urlopen(B + "/", timeout=60).read().decode("utf8")

checks = [
    ("库页 人工导入（一份谱）", "人工导入（一份谱）", lib),
    ("库页 AI 批量导入", "AI 批量导入", lib),
    ("库页 修正记录入口", "editsInfo", lib),
    ("库页 去重", "runDedupeExact", lib),
    ("库页 人工导入弹窗", "mManualBox", lib),
    ("纠错页 单击选中", "if(n){ showSel(n, d.ctrl); }", roi),
    ("纠错页 双击插入", "showIns(p.x,p.y);", roi),
    ("纠错页 Alt 拖动", "moveDrag", roi),
    ("纠错页 识别强度", 'id="stg"', roi),
    ("纠错页 局部刷新", "patchRegion", roi),
    ("纠错页 多选删除", "delSelected", roi),
]
for name, k, txt in checks:
    print("  %-22s %s" % (name, k in txt))

j = json.loads(urllib.request.urlopen(B + "/api/lib", timeout=120).read())
st = json.loads(urllib.request.urlopen(B + "/state", timeout=120).read())
print("库: %d 页 / %d 份 / 未分类 %d" % (
    len(j["items"]), len(j["groups"]),
    sum(1 for x in j["items"] if not x.get("group"))))
print("当前打开: %s，音符 %d，橙框 %d，强度 %s" % (
    st.get("title") or "(未打开)",
    len([n for n in st["notes"] if not n.get("u")]),
    len([n for n in st["notes"] if n.get("u")]), st.get("strength")))
