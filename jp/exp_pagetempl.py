"""在三月雨上验证「页内自建模板 + VLM 认代表」这条路。"""
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core, pagetempl

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"


def main():
    from jp.glyphnet import GlyphNet

    net = GlyphNet()
    pg = core.analyse(core.load_gray(P), net, path=P)
    H = pg.scale
    # 收集所有音乐行里的字形（原生尺寸）
    items = []          # (row_index, comp, patch)
    for ri, row in enumerate(pg.rows):
        if row.kind != "music":
            continue
        for c0 in row.comps:
            for c in core.split_wide_component(pg.clean, c0, H):
                if c.h >= 0.45 * H and c.area >= 8 and 0.15 <= c.aspect <= 1.15:
                    items.append((ri, c, core.patch_of(pg.clean, c, pad=0)))
    print(f"字形 {len(items)} 个（{len({i[0] for i in items})} 行）")

    patches = [p for _r, _c, p in items]
    clusters = pagetempl.cluster_patches(patches, sim=0.55)
    print(f"聚成 {len(clusters)} 类，前 10 类大小 {[len(c.members) for c in clusters[:10]]}")

    montage, _ = pagetempl._montage(patches, clusters)
    cv2.imwrite("out/templ_montage.png", montage)
    print("拼图写出 out/templ_montage.png", montage.shape)
    labels = pagetempl.ask_labels(montage)
    print("VLM 给出的类别标签:", labels)
    for i, c in enumerate(clusters[:12]):
        c.label = labels.get(str(i), "?")
        print(f"  #{i} size={len(c.members)} -> {c.label}")

    # 铺回：第一个音乐行
    rows = sorted({r for r, _c, _p in items})
    first = rows[0]
    seq = []
    for idx, (ri, c, _p) in enumerate(items):
        if ri != first:
            continue
        ci = next(i for i, cl in enumerate(clusters) if idx in cl.members)
        seq.append(clusters[ci].label if clusters[ci].label != "?" else "?")
    print("\n首行（页内模板法）:", " ".join(seq[:30]))
    print("首行（原 CNN 法）  :", " ".join(n.digit for n in pg.systems[0].notes[:30]) if pg.systems else "")


if __name__ == "__main__":
    main()
