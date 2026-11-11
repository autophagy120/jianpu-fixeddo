"""Batch entry point.

    python -m jp.run <file-or-dir> [more...] --out out/ [--settings settings.json]
                     [--set key=value ...] [--key bA] [--lyrics-mode align|group|inline|tsv]
                     [--lyrics file.txt | --lyrics-api] [--formats txt,img,json]

For every image it writes
    <name>.txt            fixed-do text score, one natural lyric phrase per line
    <name>_transposed.jpg original picture with the digits repainted in the target key
    <name>.json           QA report (per line note counts, low-confidence items, phrase source)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core, emit, preproc, score
from . import lyrics as LY
from . import music as M
from . import settings as ST

ORIG_TAG = re.compile(r"\(原调\s*([#b]?[A-Ga-g])")
PRODUCED = re.compile(r"\(原调.*转调.*\)")
VERSE_PREFIX = re.compile(r"^\s*\d+\s*[.、．]\s*")


def key_from_sibling(path: str, target: str = "C"):
    """原调常常只写在 easytr 生成的姊妹文件里；同一张图可能有多份标注（如 #F 与 G）。

    优先取“转调目标 = 本次目标”的那份，其次按文件名排序取第一个，保证可复现。
    """
    d = os.path.dirname(path) or "."
    stem = PRODUCED.sub("", os.path.basename(path))
    stem = os.path.splitext(stem)[0].strip()
    try:
        sibs = sorted(os.listdir(d))
    except OSError:
        return None
    cands = []
    for f in sibs:
        if not f.startswith(stem):
            continue
        pc = key_from_name(f)
        if pc is None:
            continue
        m = re.search(r"转调\s*([#b]?[A-Ga-g])", f)
        cands.append((pc, m.group(1) if m else ""))
    if not cands:
        return None
    for pc, tgt in cands:
        if M.parse_key(tgt) == (M.parse_key(target) or 0):
            return pc
    return cands[0][0]


def key_from_name(path: str) -> Optional[int]:
    m = ORIG_TAG.search(os.path.basename(path))
    if m:
        return M.parse_key(m.group(1))
    m = re.search(r"1\s*[=＝]\s*([#b♭♯]?[A-Ga-g])", os.path.basename(path))
    if m:
        return M.parse_key(m.group(1))
    return None


# ------------------------------------------------------------------ lyrics
def lyrics_from_api(page: core.Page, bgr: np.ndarray, model: str, verse: int = 1) -> List[str]:
    from . import vlm

    k = page.k or 1.0
    out: List[str] = []
    for sysn in page.systems:
        if not sysn.lyric_rows or verse > len(sysn.lyric_rows):
            out.append("")
            continue
        y0, y1 = sysn.lyric_rows[verse - 1]
        crop = bgr[max(0, int(y0 / k) - 8):int(y1 / k) + 8, :]
        try:
            txt = vlm.chat(
                '只输出 JSON，不要解释：{"lyrics":"这一行歌词的逐字文本（含标点，不要序号）"}。看不清用□。',
                crop, model=model, max_tokens=2500)
            txt = txt.strip()
            m = re.search(r'"lyrics"\s*:\s*"(.*?)"\s*}', txt, re.S)
            line = m.group(1) if m else txt
            line = line.replace("\\n", " ").strip()
            line = VERSE_PREFIX.sub("", line)
            out.append(line)
        except Exception as exc:
            print(f"    ! 歌词 OCR 失败 (行 {sysn.row_index}): {exc}")
            out.append("")
    return out


# ------------------------------------------------------------------ process
def process(path: str, cfg: Dict, net, out_dir: str = "", digit_clf=None) -> Dict:
    gray = core.load_gray(path)
    bgr = core.load_bgr(path)
    mode = str(cfg.get("_preproc", "off"))
    pre_info = {}
    if mode != "off":
        gray, pre_info = preproc.run(gray,
                                     do_relight=mode in ("relight", "both"),
                                     do_deskew=mode in ("deskew", "both"))
        got_angle = float(pre_info.get("angle") or 0.0)
        if pre_info.get("deskew") and abs(got_angle) >= 0.05:
            bgr = preproc.apply_to_bgr(bgr, got_angle)
    page = core.analyse(gray, net, path=path, digit_clf=digit_clf)
    key_pc = cfg["_key_override"]
    if key_pc is None and cfg["key"].get("auto_from_sibling", True):
        key_pc = key_from_sibling(path, cfg["transpose"]["target"])
    if key_pc is None and cfg["key"]["auto_from_filename"]:
        key_pc = key_from_name(path)
    rep: Dict = {"file": os.path.basename(path), "size": list(gray.shape),
                 "preproc": pre_info if pre_info else "off",
                 "systems": len(page.systems),
                 "notes": sum(len(s.notes) for s in page.systems),
                 "warnings": list(page.warnings)}
    if key_pc is None:
        rep["error"] = "未找到原调（文件名无「原调X」，请用 --key 指定）"
        return rep
    rep["key"] = M.key_name(key_pc, prefer_flat=True)
    rep["target"] = M.key_name(M.parse_key(cfg["transpose"]["target"]) or 0)

    all_pitches = [M.note_to_abs(int(n.digit), n.acc, n.octave, key_pc)
                   for s in page.systems for n in s.notes if n.digit != "0"]
    ref_cfg = cfg["fixed_do"]
    if ref_cfg["ref"] == "c":
        ref_pc, reason = 0, "强制 C 记谱"
    elif ref_cfg["ref"] == "csharp":
        ref_pc, reason = 1, "强制 C# 记谱"
    else:
        ref_pc, reason = M.pick_reference(all_pitches, int(ref_cfg["csharp_threshold"]))
    rep["ref"] = M.key_name(ref_pc)
    rep["ref_reason"] = reason
    rep["bias"] = None

    # ---- lyrics
    lyr_per_system: List[str] = []
    src = cfg["lyrics"]["source"]
    stem = os.path.splitext(os.path.basename(path))[0]
    if src == "api":
        cache = os.path.join(cfg["_out"], stem + ".lyrics.json")
        if os.path.exists(cache) and not cfg["_refresh_lyrics"]:
            lyr_per_system = json.load(open(cache, encoding="utf8"))
            rep["lyrics_source"] = "api(缓存)"
        else:
            lyr_per_system = lyrics_from_api(page, bgr, cfg["llm"]["model"])
            with open(cache, "w", encoding="utf8") as fh:
                json.dump(lyr_per_system, fh, ensure_ascii=False, indent=1)
            rep["lyrics_source"] = "api"
    elif src == "file" and cfg["lyrics"]["file"]:
        phrases = LY.load_file(cfg["lyrics"]["file"])
        lyr_per_system = LY.fit_to_systems(phrases, len(page.systems))
        rep["lyrics_source"] = "file"
    rep["lyrics"] = [l for l in lyr_per_system if l]

    # ---- render
    harm = M.HarmonicaOptions(
        breath_penalty=float(cfg["harmonica"]["breath_penalty"]),
        slide_penalty=float(cfg["harmonica"]["slide_penalty"]),
        accidental_penalty=float(cfg["harmonica"]["accidental_penalty"]),
        prefer_flat=bool(cfg["fixed_do"]["prefer_flat"]),
        allow_degree_swap=bool(cfg["harmonica"]["allow_degree_swap"]),
    )
    topts = emit.TextOptions(style=cfg["fixed_do"]["style"],
                             harmonica=bool(cfg["harmonica"]["enabled"]),
                             annotate=bool(cfg["harmonica"]["annotate"]))
    page_lines, bias = score.build_page(page, key_pc, ref_pc, harm, topts, lyr_per_system,
                                  bars_per_phrase=int(cfg["phrases"]["bars_per_phrase"]),
                                  max_per_char=int(cfg["lyrics"]["max_notes_per_char"]))
    rep["bias"] = bias
    body = score.render_page(page_lines, mode=cfg["lyrics"]["mode"],
                             continuation=cfg["lyrics"]["continuation"])

    head = [f"# {stem}",
            f"# 原调 1={M.key_name(key_pc, prefer_flat=True)}   固定调 1={M.key_name(ref_pc)}   {reason}"]
    if lyr_per_system:
        head.append(f"# 歌词来源：{rep.get('lyrics_source')}   渲染：{cfg['lyrics']['mode']}")
    if harm and topts.harmonica:
        head.append("# 半音阶口琴优化：换气 %.1f / 推键 %.1f / 升降号 %.1f"
                    % (harm.breath_penalty, harm.slide_penalty, harm.accidental_penalty))
    head.append(f"# 八度基准：以本曲中位音区为 0（bias={bias}），（ ）低八度、【 】高八度")
    head.append("")
    txt = "\n".join(head + body) + "\n"

    if "txt" in cfg["_formats"]:
        with open(os.path.join(cfg["_out"], stem + ".txt"), "w", encoding="utf8") as fh:
            fh.write(txt)
        rep["lines"] = [l for l in body if l.strip()]
        rep["phrase_sources"] = [ln.source for _si, ls in page_lines for ln in ls]

    if "img" in cfg["_formats"]:
        target_pc = M.parse_key(cfg["transpose"]["target"]) or 0
        newnotes = build_transposed(page, key_pc, target_pc,
                                    prefer_flat=bool(cfg["transpose"]["prefer_flat_spelling"]),
                                    octave_mode=str(cfg["transpose"].get("octave_mode", "keep")))
        footer = (f"原调 {M.key_name(key_pc, prefer_flat=True)} -> "
                  f"转调 {M.key_name(target_pc)}") if cfg["output"]["footer"] else ""
        iopts = emit.ImageOptions(font_path=cfg["output"]["font"],
                                  ink_color=emit.parse_color(cfg["output"].get("ink_color", "red")),
                                  footer=bool(cfg["output"]["footer"]), footer_text=footer)
        img = emit.redraw_notes(bgr, page, newnotes, iopts)
        out_img = os.path.join(cfg["_out"], stem + "_transposed.jpg")
        core.imwrite(out_img, img)
        rep["image"] = os.path.basename(out_img)

    lowconf = [(n.box, n.conf) for s in page.systems for n in s.notes if n.conf < 0.6]
    rep["low_conf_count"] = len(lowconf)
    rep["low_conf"] = [{"box": [int(v) for v in b], "conf": round(float(c), 3)} for b, c in lowconf[:40]]
    return rep


def build_transposed(page: core.Page, key_pc: int, target_pc: int, prefer_flat: bool = False,
                     octave_mode: str = "keep"):
    out: List[List[Optional[core.Note]]] = []
    for sysn in page.systems:
        row: List[Optional[core.Note]] = []
        for n in sysn.notes:
            if n.digit == "0":
                row.append(core.Note(digit="0", box=n.box, x=n.x))
                continue
            deg, acc, octv = M.transpose_movable(int(n.digit), n.acc, n.octave, key_pc, target_pc,
                                                 octave_mode=octave_mode)
            nn = core.Note(digit=str(deg), acc=acc, octave=octv, box=n.box, x=n.x,
                           acc_box=n.acc_box, dot_boxes=list(n.dot_boxes))
            nn.confirmed = getattr(n, "confirmed", True)
            row.append(nn)
        out.append(row)
    return out


def iter_images(paths: Sequence[str], cfg: Dict) -> List[str]:
    exts = {e.lower() for e in cfg["input"]["exts"]}
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() not in exts:
                        continue
                    if cfg["input"]["skip_produced"] and PRODUCED.search(f):
                        continue
                    out.append(os.path.join(root, f))
        elif os.path.isfile(p):
            out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser("jp.run")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default="", help="配置文件，默认 <项目>/settings.json")
    ap.add_argument("--set", action="append", default=[], metavar="段.键=值")
    ap.add_argument("--key", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--ref", default=None, choices=["auto", "c", "csharp"])
    ap.add_argument("--lyrics", default=None, help="歌词文本文件（每行一句）")
    ap.add_argument("--lyrics-api", action="store_true", help="用 DeepSeek 读谱面歌词")
    ap.add_argument("--lyrics-mode", default=None, choices=["align", "group", "inline", "tsv", "none"])
    ap.add_argument("--refresh-lyrics", action="store_true")
    ap.add_argument("--no-harmonica", action="store_true")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--formats", default=None)
    ap.add_argument("--font", default=None)
    ap.add_argument("--no-footer", action="store_true")
    ap.add_argument("--ink-color", default=None, help="重绘颜色 red|blue|green|black|#RRGGBB")
    ap.add_argument("--octave-mode", default=None, choices=["keep", "wrap", "absolute"])
    ap.add_argument("--preproc", default=None, choices=["off", "relight", "deskew", "both"],
                    help="参考项目的预处理；扫描件用 both，数字图建议 off")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--model", default="cnn", choices=["cnn", "templ"],
                    help="数字分类后端：cnn=合成数据 CNN；templ=语料级模板库（低分辨率页更准）")
    ap.add_argument("--all-files", action="store_true", help="连找不到原调的图也跑（默认跳过）")
    a = ap.parse_args(argv)

    cfg = ST.load(a.settings)
    cfg = ST.apply_overrides(cfg, a.set)
    if a.key:
        cfg["_key_override"] = M.parse_key(a.key)
    else:
        cfg["_key_override"] = None
    if a.target:
        cfg["transpose"]["target"] = a.target
    if a.ref:
        cfg["fixed_do"]["ref"] = a.ref
    if a.lyrics:
        cfg["lyrics"]["source"] = "file"
        cfg["lyrics"]["file"] = a.lyrics
    if a.lyrics_api:
        cfg["lyrics"]["source"] = "api"
    if a.lyrics_mode:
        cfg["lyrics"]["mode"] = a.lyrics_mode
    if a.no_harmonica:
        cfg["harmonica"]["enabled"] = False
    if a.annotate:
        cfg["harmonica"]["annotate"] = True
    if a.formats:
        cfg["output"]["formats"] = [x.strip() for x in a.formats.split(",") if x.strip()]
    if a.font:
        cfg["output"]["font"] = a.font
    if a.no_footer:
        cfg["output"]["footer"] = False
    if a.ink_color:
        cfg["output"]["ink_color"] = a.ink_color
    if a.octave_mode:
        cfg["transpose"]["octave_mode"] = a.octave_mode
    cfg["_preproc"] = a.preproc or "off"
    cfg["_formats"] = tuple(cfg["output"]["formats"])
    cfg["_refresh_lyrics"] = a.refresh_lyrics
    cfg["_only_keyed"] = not a.all_files

    os.makedirs(a.out, exist_ok=True)
    ST.save(cfg, a.settings)          # 把生效的配置存下来，便于下次改
    net = None
    if not a.no_model:
        from .glyphnet import GlyphNet
        net = GlyphNet()
    digit_clf = None
    if a.model == "templ":
        from .templclass import make as make_templ
        digit_clf = make_templ()

    files = iter_images(a.paths, cfg)
    if cfg["_key_override"] is None and not a.all_files:
        keyed = [f for f in files if key_from_sibling(f, cfg["transpose"]["target"]) is not None or key_from_name(f) is not None]
        print(f"跳过 {len(files) - len(keyed)} 张（找不到原调，无姊妹标注文件）")
        files = keyed
    print(f"{len(files)} 张图  |  固定调 {cfg['fixed_do']['ref']}  |  歌词 {cfg['lyrics']['source']}/"
          f"{cfg['lyrics']['mode']}  |  口琴 {cfg['harmonica']['enabled']}", flush=True)
    roots = [p for p in a.paths if os.path.isdir(p)]
    reports = []
    for i, f in enumerate(files, 1):
        out_dir = a.out
        for root in roots:
            if os.path.abspath(f).startswith(os.path.abspath(root)):
                rel = os.path.dirname(os.path.relpath(f, root))
                if rel and rel != ".":
                    out_dir = os.path.join(a.out, rel)
                break
        os.makedirs(out_dir, exist_ok=True)
        cfg["_out"] = out_dir
        try:
            r = process(f, cfg, net, out_dir, digit_clf=digit_clf)
        except Exception as exc:
            r = {"file": os.path.basename(f), "error": f"{type(exc).__name__}: {exc}"}
        reports.append(r)
        flag = "OK " if "error" not in r else "ERR"
        print(f"[{i}/{len(files)}] {flag} {r['file'][:56]:56s} 行={r.get('systems', '-')} "
              f"音符={r.get('notes', '-')} 低置信={r.get('low_conf_count', '-')} {r.get('error', '')}", flush=True)
    with open(os.path.join(a.out, "_qa.json"), "w", encoding="utf8") as fh:
        json.dump(reports, fh, ensure_ascii=False, indent=1)
    ok = sum(1 for r in reports if "error" not in r)
    print(f"完成 {ok}/{len(reports)}，报告 {os.path.join(a.out, '_qa.json')}")


if __name__ == "__main__":
    main()
