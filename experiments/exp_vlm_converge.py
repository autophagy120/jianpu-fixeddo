"""诊断：同一片段在不同缩放/多次采样下，VLM 能否收敛到一致结果。"""
import sys
sys.path.insert(0, ".")
import cv2
from collections import Counter

from jp import core
import jp.exp_vlm_str as V

P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"

CASES = [
    ("片1 x=0..280", 0, 280),
    ("片2 x=220..500", 220, 500),
    ("片3 x=440..720", 440, 720),
]


def main():
    bgr = core.load_bgr(P)
    for tag, x0, x1 in CASES:
        print(f"\n=== {tag} ===")
        for scale in (4.0, 5.0, 6.0):
            crop = bgr[126:172, x0:x1]
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            seqs = []
            for _ in range(5):
                try:
                    seqs.append(V.ask(big, temperature=0.4))
                except Exception:
                    seqs.append([])
            lens = [len(s) for s in seqs]
            common = Counter(lens).most_common(1)[0]
            same = [s for s in seqs if len(s) == common[0]]
            agree = None
            if same:
                toks = [Counter(s[i] for s in same).most_common(1)[0][0] for i in range(len(same[0]))]
                agree = " ".join(toks)
            print(f"  scale={scale}: 长度 {lens}  众数 {common[0]}x{common[1]}  "
                  f"投票: {(agree or '')[:90]}")


if __name__ == "__main__":
    main()
