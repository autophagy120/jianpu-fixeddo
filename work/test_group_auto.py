"""测自动分组：时间间隔 + 歌词相似度。合成两份谱各 3 页。"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jp.library as L
from PIL import Image

TMP = os.path.join(L.ROOT, "work", "_group_test")
os.makedirs(TMP, exist_ok=True)
base = int(time.time()) - 10000

LYR_A = "我们一起唱这首歌吧 啦啦啦 风轻轻吹过 天空很蓝 你的笑容像阳光"
LYR_B = "深夜里独自走在街头 灯光闪烁 雨还在下 没有人回答 我一个人的歌"


def mk(idx, t_offset, text):
    img = os.path.join(TMP, "p%d.png" % idx)
    if not os.path.exists(img):
        Image.new("RGB", (40, 30), (255, 255, 255)).save(img)
    txt = os.path.join(TMP, "p%d.txt" % idx)
    with open(txt, "w", encoding="utf8") as f:
        f.write(text or "")
    return dict(id="t%02d" % idx, song="测试份", image=img, origin=img,
                title="", ocr_txt=txt, key="", target="C", status="ocr",
                out="", notes=0, uncovered=0, keys=[], multi=False, dup=False,
                order=idx, t0=base + t_offset, t1=base + t_offset,
                updated=time.strftime("%Y-%m-%d %H:%M:%S"))


def run_case(name, texts, offsets, expect_n):
    items = L.load()
    for it in list(items.values()):
        if it["id"].startswith("t0") and it.get("song") == "测试份":
            del items[it["id"]]
    for k, (txt, off) in enumerate(zip(texts, offsets), 1):
        e = mk(k, off, txt)
        items[e["id"]] = e
    L.save(items)
    ids = ["t%02d" % k for k in range(1, len(texts) + 1)]
    res = L.auto_group(ids=ids, log=lambda s: None)
    items = L.load()
    groups = {}
    for i in ids:
        groups.setdefault(items[i].get("group"), []).append(i)
    print("%-28s → %d 份: %s" % (name, len(groups),
                                 [sorted(v) for v in groups.values()]))
    ok = len(groups) == expect_n
    print("   ", "PASS" if ok else "FAIL（期望 %d 份）" % expect_n)
    return ok


allok = True
# 情况1：有歌词 —— 虽然时间间隔很大，歌词一样就算同一份
allok &= run_case("歌词相同 + 大间隔",
                  [LYR_A] * 3 + [LYR_B] * 3,
                  [0, 600, 1200, 2400, 3000, 3600], 2)

# 情况2：无歌词（谱面文字为空）—— 只能靠时间间隔：5s 内连续，20s 断开
allok &= run_case("无歌词 + 5s/20s 间隔",
                  ["", "", "", "", "", ""],
                  [0, 5, 10, 30, 35, 40], 2)

# 情况3：无歌词，全部很接近 —— 应当判成一份
allok &= run_case("无歌词 + 全部 4s 间隔",
                  ["", "", "", "", "", ""],
                  [0, 4, 8, 12, 16, 20], 1)

# 情况4：歌词不同（两份不同的歌）+ 时间也断 → 两份
allok &= run_case("歌词不同 + 大间隔",
                  [LYR_A, LYR_A, LYR_A, LYR_B, LYR_B, LYR_B],
                  [0, 5, 10, 40, 45, 50], 2)

# 清理
items = L.load()
for it in list(items.values()):
    if it.get("song") == "测试份":
        del items[it["id"]]
L.save(items)
import shutil
shutil.rmtree(TMP, ignore_errors=True)
print("\n清理后条目:", len(L.load()))
print("RESULT:", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
