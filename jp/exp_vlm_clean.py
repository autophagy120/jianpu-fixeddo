"""把谱行里的"非音符结构"（小节线、反复点、减时线、括号）涂白后再交给 VLM。

思路：这些符号会让模型乱，而我的几何检测本来就知道它们在哪儿。
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core, vlm, vlmline
from collections import Counter




def structure_mask(gray: np.ndarray, h_med: float) -> np.ndarray:
    """返回需要涂白的掩码（画成 255 的地方会被抹掉）。"""
    bw = core.binarize(gray)
    hz = max(8, int(0.55 * h_med))      # 减时线/延音线：细长横线
    vt = max(8, int(0.9 * h_med))       # 小节线/反复竖线：细长竖线
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hz, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vt)))
    m = cv2.bitwise_or(hor, ver)
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


def clean_band(bgr: np.ndarray, y0: int, y1: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    comps = core.components(core.binarize(gray), min_area=4)
    hs = [c.h for c in comps if c.h >= 6 and 0.3 <= c.aspect <= 1.0]
    h_med = float(np.median(hs)) if hs else 12.0
    m = structure_mask(gray, h_med)[y0:y1]
    band = bgr[y0:y1].copy()
    band[m > 0] = 255
    return band, h_med


def read_clean(bgr, y0, y1, x0, x1, scale, samples=5):
    band, h_med = clean_band(bgr, y0, y1)
    crop = band[:, x0:x1]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    seqs = []
    for _ in range(samples):
        try:
            seqs.append(vlm.read_note_string(big, temperature=0.4))
        except Exception:
            seqs.append([])
    seqs = [s for s in seqs if s]
    if not seqs:
        return [], 0.0
    lens = Counter(len(s) for s in seqs)
    n_mode, cnt = lens.most_common(1)[0]
    same = [s for s in seqs if len(s) == n_mode]
    toks = [Counter(s[i] for s in same).most_common(1)[0][0] for i in range(n_mode)]
    per_pos = min(Counter(s[i] for s in same).most_common(1)[0][1] / len(same) for i in range(n_mode))
    return toks, cnt / len(seqs) * per_pos


def main():
    bgr = core.load_bgr(r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg")
    band, h_med = clean_band(bgr, 126, 172)
    print(f"h_med={h_med:.1f}")
    cv2.imwrite("out/band_cleaned.png",
                cv2.resize(band, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    print("写出 out/band_cleaned.png")
    for x0, x1 in ((280, 560), (240, 520), (300, 580)):
        for scale in (6.0, 5.0):
            toks, sup = read_clean(bgr, 126, 172, x0, x1, scale)
            print(f"  x={x0}..{x1} scale={scale}: 支持 {sup:.2f}  {' '.join(toks[:16])}")


if __name__ == "__main__":
    main()
