"""Exploratory probe: what does a real score look like after binarisation?"""
from __future__ import annotations

import sys

import cv2
import numpy as np


def load(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def main(path):
    g = load(path)
    print("shape", g.shape, "min/max", g.min(), g.max())
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("ink ratio", round(float((bw > 0).mean()), 4))

    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    rules = cv2.bitwise_or(hor, ver)
    clean = cv2.bitwise_and(bw, cv2.bitwise_not(rules))
    print("rule pixels", int((rules > 0).sum()), "clean ink", int((clean > 0).sum()))

    prof = clean.astype(np.float64).sum(axis=1) / 255
    nz = np.nonzero(prof > 0.5)[0]
    print("ink rows", nz[0], "..", nz[-1])
    runs = []
    s = None
    for i, v in enumerate(prof > 0.5):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(prof)))
    print(f"{len(runs)} raw bands")
    n, labels, stats, cent = cv2.connectedComponentsWithStats(clean, connectivity=8)
    areas = stats[1:, 4]
    hs = stats[1:, 3]
    print(f"{n-1} components; height median={np.median(hs):.0f} p90={np.percentile(hs,90):.0f} max={hs.max()}")
    for b in runs:
        sub = clean[b[0]:b[1]]
        cols = np.nonzero(sub.sum(axis=0) > 0)[0]
        nn, _, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        hh = st[1:, 3] if nn > 1 else np.array([0])
        print(f"  band y={b[0]:4d}-{b[1]:4d} h={b[1]-b[0]:3d} x={cols[0]:4d}..{cols[-1]:4d} "
              f"comps={nn-1:3d} h_med={np.median(hh):5.1f} h_max={hh.max():3d}")


if __name__ == "__main__":
    main(sys.argv[1])
