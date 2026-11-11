"""页内自建模板 v2：聚类 -> VLM 标代表 -> 全局最优分配（带 margin）。"""
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core, pagetempl

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
GT_FIRST = ("6 1 2 3 " * 6)  # 第 1 行是 6123 循环（真值）


def main():
    from jp.glyphnet import GlyphNet

    net = GlyphNet()
    pg = core.analyse(core.load_gray(P), net, path=P)
    H = pg.scale
    items = []
    for ri, row in enumerate(pg.rows):
        if row.kind != "music":
            continue
        for c0 in row.comps:
            for c in core.split_wide_component(pg.clean, c0, H):
                if c.h >= 0.45 * H and c.area >= 8 and 0.15 <= c.aspect <= 1.15:
                    items.append((ri, c, core.patch_of(pg.clean, c, pad=0)))
    patches = [p for _r, _c, p in items]
    print(f"字形 {len(patches)}")

    clusters = pagetempl.cluster_patches(patches, sim=0.55)
    montage, _ = pagetempl._montage(patches, clusters)
    cv2.imwrite("out/templ_montage.png", montage)
    labels = pagetempl.label_clusters(patches, clusters, chunk=6, debug=True)
    for i, c in enumerate(clusters):
        c.label = labels.get(i, "?")
    print("类别标签:", [(i, len(c.members), c.label) for i, c in enumerate(clusters)])

    usable = [i for i, c in enumerate(clusters) if c.label in "01234567"]
    labs, confs = pagetempl.assign_by_centroids(patches, clusters, usable)

    # 第一行
    rows = sorted({r for r, _c, _p in items})
    first = rows[0]
    seq_pt, seq_cnn, gt = [], [], []
    for (ri, c, p), lab in zip(items, labs):
        if ri != first:
            continue
        seq_pt.append(lab or "?")
        match = [n for n in pg.systems if n.row_index >= 0]
    # CNN 结果（用于对比）
    cnn = []
    for sysn in pg.systems:
        if sysn.notes and abs(sysn.y0 - pg.rows[first].y0) < 2:
            cnn = [n.digit for n in sysn.notes]
            break

    n = min(len(seq_pt), len(gt))
    print("\n首行 页内模板:", " ".join(seq_pt[:30]))
    print("首行 CNN     :", " ".join(cnn[:30]))
    got = [x for x in seq_pt if x != "?"]
    print(f"覆盖率 {len(got)}/{len(seq_pt)}  (未定 {seq_pt.count('?')})")


if __name__ == "__main__":
    main()
