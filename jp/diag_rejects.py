"""诊断：谱行里被判成 'other' 的候选，到底是什么样子。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np

from jp import core
from jp.glyphnet import GlyphNet


def main(path, row_idx=2):
    net = GlyphNet()
    g = core.load_gray(path)
    pg = core.analyse(g, net, path=path)
    music_rows = [r for r in pg.rows if r.kind == "music"]
    print(f"scale={pg.scale:.1f} k={pg.k:.2f} rows={len(pg.rows)} music_rows={len(music_rows)}")
    r = music_rows[min(row_idx, len(music_rows) - 1)]
    H = pg.scale
    comps = [c for c in r.comps if c.h >= 0.5 * H]
    print(f"music row y={r.y0}-{r.y1} comps>=0.5H: {len(comps)}")
    patches = [core.patch_of(pg.clean, c) for c in comps]
    cls, prob = net.predict(patches)
    tiles, labels = [], []
    rej = 0
    for c, p, k, pb in zip(comps, patches, cls, prob):
        name = net.classes[int(k)]
        if name not in "01234567":
            rej += 1
        t = cv2.resize(p, (48, 48), interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        t = cv2.copyMakeBorder(t, 16, 4, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cv2.putText(t, f"{name[:2]}{pb:.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (0, 255, 0) if name in "01234567" else (0, 0, 255), 1)
        tiles.append(t)
    print(f"判为数字 {len(comps) - rej} / {len(comps)}")
    per = 24
    for i in range(0, len(tiles), per):
        cv2.imwrite(f"out/diag_row_{row_idx}_{i // per}.png", np.hstack(tiles[i:i + per]))
    print("写出 out/diag_row_%d_*.png" % row_idx)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
