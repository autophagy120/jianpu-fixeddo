"""把人的修正记下来，攒成自学习数据集。

为什么单独一个模块：修正数据是**最贵的标注**（一张谱人工改 10 个音，等于
10 个高置信标签 + 10 个错误样本），比 `train_glyph.labels_from_page()` 那种
"按小节自动对齐猜标签"可靠得多。这里只负责"如实记录"，训练放到
`jp/train_glyph.py` / `work/harvest_edits.py`。

目录：
    work/train/events.jsonl      所有操作事件（含未产生字形样本的，如改字号）
    work/train/index.jsonl       每个字形样本一行（patch 文件 + 标签 + 出处）
    work/train/patches/*.npy     32x32 float32 字形图（从 page.clean 抠，和训练同域）
    library/<id>/edits.jsonl     同样的事件，按页留一份，便于单页复查

标签约定：
    数字 '0'..'7'   这是这个音的字形
    'reject'        这里不是音符（删除/清除）
    'blank'         这里是空白（加强清除擦掉的区域）
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "work", "train")
PATCH_DIR = os.path.join(TRAIN, "patches")
EVENTS = os.path.join(TRAIN, "events.jsonl")
INDEX = os.path.join(TRAIN, "index.jsonl")
PATCH = 32

_seq = [0]


def _ensure():
    os.makedirs(PATCH_DIR, exist_ok=True)


def enabled() -> bool:
    """总开关（设置里可关）：默认开。"""
    p = os.path.join(TRAIN, "disabled")
    return not os.path.exists(p)


def set_enabled(on: bool):
    _ensure()
    p = os.path.join(TRAIN, "disabled")
    if on:
        if os.path.exists(p):
            os.remove(p)
    else:
        open(p, "w").write("1")


def next_seq() -> int:
    _seq[0] += 1
    return _seq[0]


def cur_seq() -> int:
    return _seq[0]


def _append(path: str, obj: dict):
    _ensure()
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_event(page_id: str, kind: str, **kw):
    """记一条操作事件（不产生训练样本的操作也记，方便还原人的操作序列）。"""
    if not enabled():
        return None
    seq = next_seq()
    rec = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), page=page_id or "", kind=kind,
               seq=seq, **kw)
    _append(EVENTS, rec)
    if page_id:
        d = os.path.join(ROOT, "library", page_id)
        if os.path.isdir(d):
            _append(os.path.join(d, "edits.jsonl"), rec)
    return seq


def patch_from_clean(clean: np.ndarray, box, size: int = PATCH) -> np.ndarray:
    """按框从二值图抠一个 size×size 的字形图（复用 train_glyph 的预处理）。"""
    from .train_glyph import patch_of_box
    return patch_of_box(clean, box, size)


def has_ink(patch: np.ndarray, thresh: float = 6.0) -> bool:
    return float(patch.sum()) >= thresh


def add_sample(patch: np.ndarray, label: str, page_id: str, box,
               pred: str = "", kind: str = "", extra: Optional[dict] = None,
               ctx: Optional[dict] = None) -> Optional[str]:
    """存一个字形样本，返回文件名。`label` 见模块头。"""
    if not enabled():
        return None
    _ensure()
    seq = next_seq()
    key = "%s|%s|%s|%s|%d" % (page_id, label, ",".join("%.0f" % v for v in box),
                             pred, seq)
    name = hashlib.sha1(key.encode("utf8")).hexdigest()[:16] + ".npy"
    np.save(os.path.join(PATCH_DIR, name), patch.astype(np.float32))
    rec = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), page=page_id or "", kind=kind,
               seq=int(seq), label=str(label), pred=str(pred), file=name,
               box=[float(v) for v in box])
    if ctx:
        rec["ctx"] = ctx
    if extra:
        rec.update(extra)
    _append(INDEX, rec)
    return name


# ---------------------------------------------------------------- 读回
def load_events() -> list:
    out = []
    if os.path.exists(EVENTS):
        for ln in open(EVENTS, encoding="utf8"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def load_index() -> list:
    out = []
    if os.path.exists(INDEX):
        for ln in open(INDEX, encoding="utf8"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def undone_ranges(events: list) -> dict:
    """把 undo 事件翻成 {page: [(seq_from, seq_to)]}，用来过滤被撤销的样本。"""
    per = {}
    for e in events:
        if e.get("kind") != "undo":
            continue
        page = e.get("page") or ""
        a, b = int(e.get("seq_from") or 0), int(e.get("seq_to") or 0)
        if a:
            per.setdefault(page, []).append((a, b or 10 ** 9))
    return per


def load_dataset(drop_undone: bool = True):
    """返回 (patches[N,32,32], labels[N] str, meta list)。已被撤销的样本默认丢掉。"""
    idx = load_index()
    if drop_undone:
        ranges = undone_ranges(load_events())
        kept = []
        for r in idx:
            page = r.get("page") or ""
            seq = int(r.get("seq") or 0)
            drop = any(a < seq <= b for (a, b) in ranges.get(page, []))
            if not drop:
                kept.append(r)
        idx = kept
    xs, ys = [], []
    for r in idx:
        p = os.path.join(PATCH_DIR, r.get("file") or "")
        if not os.path.exists(p):
            continue
        try:
            xs.append(np.load(p))
        except Exception:
            continue
        ys.append(str(r.get("label")))
    if not xs:
        return np.zeros((0, PATCH, PATCH), np.float32), [], idx
    return np.stack(xs), ys, idx


def stats() -> dict:
    idx = load_index()
    ev = load_events()
    from collections import Counter
    by_label = Counter(str(r.get("label")) for r in idx)
    by_kind = Counter(str(r.get("kind")) for r in idx)
    by_event = Counter(str(e.get("kind")) for e in ev)
    return dict(samples=len(idx), events=len(ev), enabled=enabled(),
                by_label=dict(by_label.most_common(12)),
                by_kind=dict(by_kind.most_common(12)),
                by_event=dict(by_event.most_common(12)))
