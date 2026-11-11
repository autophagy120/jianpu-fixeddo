"""把人工修正样本汇总成训练集，并给出「自学习」现状报告。

用法：
    python work\\harvest_edits.py            # 只统计 + 导出 npz
    python work\\harvest_edits.py --train    # 顺带在人工样本上跑一遍训练（小样本仅验证流程）
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, r"D:\简谱转调项目")
import jp.edits as ED

ap = argparse.ArgumentParser()
ap.add_argument("--train", action="store_true")
ap.add_argument("--out", default=os.path.join(ED.TRAIN, "manual_dataset.npz"))
args = ap.parse_args()

st = ED.stats()
print("修正记录：")
print("  事件 %d 条，字形样本 %d 个，开关=%s" % (st["events"], st["samples"], st["enabled"]))
print("  按标签：", st["by_label"])
print("  按来源：", st["by_kind"])
print("  按操作：", st["by_event"])

xs, ys, meta = ED.load_dataset()
print("\n未被撤销的样本：%d 个" % len(xs))
if len(xs):
    c = Counter(ys)
    print("  标签分布：", dict(c.most_common()))
    np.savez_compressed(args.out, X=xs, y=np.array([int(y) if y.isdigit() else -1 for y in ys]))
    print("  已导出：", args.out)

if args.train and len(xs) >= 40:
    import jp.train_glyph as TG
    ys_num = [int(y) for y in ys if y.isdigit()]
    xs_num = [x for x, y in zip(xs, ys) if y.isdigit()]
    print("\n用 %d 个人工样本训练（仅验证流程，样本还太少）：" % len(xs_num))
    model = TG.train(xs_num, ys_num, epochs=20)
    print("  训练集准确率：%.3f" % TG.accuracy(model, xs_num, ys_num))
elif args.train:
    print("\n样本太少（<40），先不训练；继续改谱就会自动攒。")
