"""超分实验：把 9px 的页面放大 4 倍再识别，看数字是否变对。"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core
from jp import compare

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"


def sr(gray, model="FSRCNN_x4", path=r"models\sr\FSRCNN_x4.pb"):
    s = cv2.dnn_superres.DnnSuperResImpl_create()
    s.readModel(path)
    s.setModel("fsrcnn", 4)
    return s.upsample(gray)


def main():
    from jp.glyphnet import GlyphNet

    net = GlyphNet()
    g = core.load_gray(P)
    print("原图", g.shape)

    for name, model, path in (("FSRCNN_x4", "fsrcnn", r"models\sr\FSRCNN_x4.pb"),
                              ("ESPCN_x4", "espcn", r"models\sr\ESPCN_x4.pb")):
        try:
            up = sr(g, model=model, path=path)
        except Exception as exc:
            print(f"{name} 失败: {exc}")
            continue
        print(f"{name} -> {up.shape}")
        core.imwrite(f"out/sr_{name}.png", up)
        pg = core.analyse(up, net, path=name)
        tot = sum(len(s.notes) for s in pg.systems)
        low = sum(1 for s in pg.systems for n in s.notes if n.conf < 0.6)
        m = [s for s in pg.systems if s.notes]
        print(f"  行={len(pg.systems)} 音符={tot} 低置信={low}")
        if m:
            print("  首行:", " ".join(compare.token_of(n) for n in m[0].notes[:20]))


if __name__ == "__main__":
    main()
