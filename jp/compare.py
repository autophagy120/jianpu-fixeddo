"""我的转调图 vs easytr 转调图：三图对照 + 互校验.

没有金标准，就用"两条独立的识别链路互相印证"：
  A = 原图识别 → 我的转调结果（渲染成图后再识别回来）
  B = easytr 的转调图直接识别
两者都在目标调上，差异处必然是"easytr 识别错了 / 它渲染错了 / 我识别错了 / 我渲染错了"之一，
逐行列出差异，人工只看差异处即可。
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jp import core, emit  # noqa: E402
from jp import music as M  # noqa: E402
from jp import music as M  # noqa: E402

PRODUCED = re.compile(r"\(原调\s*([#b]?[A-Ga-g])\s*-\s*转调\s*([#b]?[A-Ga-g])\)")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def token_of(n: core.Note) -> str:
    octv = n.octave
    tail = ""
    if octv > 0:
        tail = "^" * min(octv, 2)
    elif octv < 0:
        tail = "v" * min(-octv, 2)
    return f"{n.acc}{n.digit}{tail}"


def recognise_flat(path: str, net):
    """返回 (逐行 token 列表, 展平后的 token 列表)。比较要用展平的，按行比会整行判错。"""
    gray = core.load_gray(path)
    page = core.analyse(gray, net, path=path)
    lines: List[str] = []
    flat: List[str] = []
    for sysn in page.systems:
        if not sysn.notes:
            continue
        toks = [token_of(n) for n in sysn.notes]
        lines.append(" ".join(toks))
        flat.extend(toks)
    return lines, flat


def recognise_tokens(path: str, net) -> List[str]:
    gray = core.load_gray(path)
    page = core.analyse(gray, net, path=path)
    out: List[str] = []
    for sysn in page.systems:
        if not sysn.notes:
            continue
        out.append(" ".join(token_of(n) for n in sysn.notes))
    return out


def agreement(a: Sequence[str], b: Sequence[str]) -> Tuple[float, List[str]]:
    if not a and not b:
        return 1.0, []
    sa, sb = " | ".join(a), " | ".join(b)
    sm = difflib.SequenceMatcher(None, a, b)
    diffs: List[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        diffs.append(f"{op}: 我的[{i1}:{i2}]={' '.join(a[i1:i2])[:80]}  easytr[{j1}:{j2}]={' '.join(b[j1:j2])[:80]}")
    return sm.ratio(), diffs


def three_panel(org: np.ndarray, mine: np.ndarray, ref: np.ndarray, H: int = 1400) -> np.ndarray:
    def fit(im):
        w = max(1, int(round(im.shape[1] * H / im.shape[0])))
        return im if im.shape[0] == H and im.shape[1] == w else __import__("cv2").resize(
            im, (w, H), interpolation=__import__("cv2").INTER_AREA)

    import cv2
    a, b, c = fit(org), fit(mine), fit(ref)
    gap = np.full((H, 16, 3), 80, np.uint8)
    return np.hstack([a, gap, b, gap, c])


def find_pairs(root: str) -> List[Dict]:
    """配对原图与 easytr 成品。

    原调以 run.key_from_sibling 的判定为准（同一首歌可能有两份标注，如 #F 与 G），
    再挑“原调与它一致”的那份成品，保证两边比的是同一次转调。
    """
    from . import run as RUN

    pairs: List[Dict] = []
    for dirpath, _dirs, files in os.walk(root):
        produced: Dict[str, List] = {}
        originals: Dict[str, str] = {}
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            m = PRODUCED.search(f)
            if m:
                stem = os.path.splitext(PRODUCED.sub("", f))[0].strip()
                produced.setdefault(stem, []).append((f, m.group(1), m.group(2)))
            else:
                originals[os.path.splitext(f)[0]] = f
        for stem, cands in produced.items():
            if stem not in originals:
                continue
            orig = os.path.join(dirpath, originals[stem])
            key_pc = RUN.key_from_sibling(orig, "C")
            pick = None
            for c in cands:
                if M.parse_key(c[1]) == key_pc:
                    pick = c
                    break
            pick = pick or cands[0]
            pairs.append({"dir": dirpath, "song": os.path.basename(dirpath), "stem": stem,
                          "orig": orig, "ref": os.path.join(dirpath, pick[0]),
                          "key": pick[1], "target": pick[2], "key_pc": key_pc,
                          "n_annotations": len(cands)})
    return pairs


def main():
    from jp.glyphnet import GlyphNet

    root = sys.argv[1] if len(sys.argv) > 1 else r"E:\DeepSeek-Harness\工作目录\测试\转调简谱"
    batch = sys.argv[2] if len(sys.argv) > 2 else os.path.join("out", "batch")
    outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join("out", "compare")
    os.makedirs(outdir, exist_ok=True)
    net = GlyphNet()

    pairs = find_pairs(root)
    print(f"{len(pairs)} 组（原图 + easytr 成品）", flush=True)
    rows = []
    for i, p in enumerate(pairs, 1):
        rel = os.path.relpath(p["dir"], root)
        mine_path = os.path.join(batch, rel, p["stem"] + "_transposed.jpg")
        rec = {"song": p["song"], "stem": p["stem"], "key": p["key"], "target": p["target"]}
        if not os.path.exists(mine_path):
            rec["error"] = "我的产物缺失"
            rows.append(rec)
            print(f"[{i}/{len(pairs)}] 缺 {p['song']}", flush=True)
            continue
        try:
            from . import run as RUN
            target_pc = M.parse_key(p["target"]) or 0
            page = core.analyse(core.load_gray(p["orig"]), net, path=p["orig"])
            new = RUN.build_transposed(page, p["key_pc"], target_pc)
            lines_mine = [" ".join(token_of(n) for n in row if n is not None) for row in new if row]
            toks_mine = [token_of(n) for row in new for n in row if n is not None]
            lines_ref, toks_ref = recognise_flat(p["ref"], net)
            ratio, diffs = agreement(toks_mine, toks_ref)
            rec.update({"ratio": round(ratio, 4), "lines_mine": len(lines_mine),
                        "lines_ref": len(lines_ref), "tokens_mine": len(toks_mine),
                        "tokens_ref": len(toks_ref), "diffs": diffs[:12]})
            org = core.load_bgr(p["orig"])
            mine = core.load_bgr(mine_path)
            ref = core.load_bgr(p["ref"])
            sheet = three_panel(org, mine, ref)
            name = re.sub(r'[\\/:*?"<>|]', "_", p["song"])[:60]
            out_png = os.path.join(outdir, f"{i:02d}_{name}.jpg")
            core.imwrite(out_png, sheet)
            rec["sheet"] = os.path.basename(out_png)
            print(f"[{i}/{len(pairs)}] {p['song'][:20]:20s} 一致度 {ratio:.3f} "
                  f"行 {len(toks_mine)}/{len(toks_ref)}", flush=True)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{i}/{len(pairs)}] ERR {p['song'][:20]} {exc}", flush=True)
        rows.append(rec)

    with open(os.path.join(outdir, "_agreement.json"), "w", encoding="utf8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    # index.html
    html = ["<meta charset=utf-8><title>转调对照</title>",
            "<style>body{font-family:system-ui;margin:20px}td{padding:4px 10px;border-bottom:1px solid #eee}"
            "img{width:100%;max-width:1400px;border:1px solid #ccc}.lo{color:#c00}.ok{color:#070}</style>",
            f"<h2>转调对照（{len(rows)} 组）：原图 | 我的 | easytr</h2>",
            "<table><tr><th>#</th><th>歌曲</th><th>原调→目标</th><th>一致度</th><th>图</th></tr>"]
    for i, r in enumerate(rows, 1):
        ratio = r.get("ratio")
        cls = "ok" if (ratio or 0) >= 0.95 else "lo"
        html.append(f"<tr><td>{i}</td><td>{r['song']}</td><td>1={r['key']} → {r['target']}</td>"
                    f"<td class={cls}>{'' if ratio is None else f'{ratio:.1%}'}</td>"
                    f"<td>{r.get('sheet', r.get('error', ''))}</td></tr>")
    html.append("</table>")
    for i, r in enumerate(rows, 1):
        if r.get("sheet"):
            html.append(f"<h3>{i}. {r['song']} <small>{r['key']}→{r['target']} 一致度 "
                        f"{r['ratio']:.1%}</small></h3><img src='{r['sheet']}'>")
            if r.get("diffs"):
                html.append("<pre>" + "\n".join(r["diffs"]) + "</pre>")
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf8") as fh:
        fh.write("\n".join(html))
    ok = [r for r in rows if r.get("ratio") is not None]
    if ok:
        avg = sum(r["ratio"] for r in ok) / len(ok)
        print(f"\n平均一致度 {avg:.3f}（{len(ok)}/{len(rows)} 组）  报告 {outdir}/_agreement.json  "
              f"浏览 {outdir}/index.html")


if __name__ == "__main__":
    main()
