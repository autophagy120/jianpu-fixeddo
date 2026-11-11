"""在三月雨上验证：小节线锚定 + VLM 补数字。"""
import os
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, ".")
import numpy as np

from jp import core
from jp import phrases as PH
from jp import vlmfill
from jp.glyphnet import GlyphNet

p = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"


def main():
    net = GlyphNet()
    pg = core.analyse(core.load_gray(p), net, path=p)
    bgr = core.load_bgr(p)
    print(f"scale={pg.scale:.1f} k={pg.k:.2f} systems={len(pg.systems)}")
    tot = rep_ok = 0
    for si, sysn in enumerate(pg.systems[:1]):
        if not sysn.notes:
            continue
        before = " ".join(f"{n.digit}{'^' * max(0, n.octave)}{'v' * max(0, -n.octave)}"
                          for n in sysn.notes[:14])
        vn = vlmfill.vlm_digits_for_line(bgr, pg, sysn)
        bars = PH.bar_positions(pg, sysn, pg.scale)
        rep = vlmfill.fill_line(pg, sysn, vn, bars)
        after = " ".join(f"{n.acc}{n.digit}{'^' * max(0, n.octave)}{'v' * max(0, -n.octave)}"
                         for n in sysn.notes[:14])
        tot += 1
        rep_ok += 1 if rep.matched and not rep.flagged_bars else 0
        print(f"\n行{si} y={sysn.y0:.0f} 小节={rep.bars} VLM={len(vn)} 音符={len(sysn.notes)} "
              f"匹配={rep.matched} 多={rep.extra_from_vlm} 少={rep.missing_from_vlm} 待核对小节={rep.flagged_bars}")
        print("  前: ", before)
        print("  后: ", after)


if __name__ == "__main__":
    main()
