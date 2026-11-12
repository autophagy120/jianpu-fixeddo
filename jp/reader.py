"""演奏阅读：只读、预渲染、多页并排。

为什么单独一层：演奏时不能等渲染。这里把「整份 × 目标调」的图片提前渲染好并落盘，
阅读页只拿现成 JPEG，翻页是纯前端切图，零等待。
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import emit_fixed as EF
from . import library as LIB
from . import music as M

# 笛子：筒音做 X 时，主音 1 相对筒音的音程（半音，向下为正）
DEG_SEMI = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}


def tongyin_target(dizi_key: str, tongyin: int) -> str:
    """𥱊调 X 筒音做 N 指法 → 实际演奏调（1=?）。

    例：G 调笛筒音做 5 → 1=C；筒音做 2 → 1=F。
    """
    pc = M.parse_key(dizi_key)
    if pc is None:
        return ""
    semi = DEG_SEMI.get(int(tongyin), 0)
    target = (int(pc) - semi) % 12
    return M.key_name(target)


def _cache_mtime(pid: str) -> float:
    p = os.path.join(LIB.LIB_DIR, pid, "cache.pkl")
    try:
        return os.path.getmtime(p)
    except Exception:
        return 0.0


def _read_path(pid: str, target_pc: int) -> str:
    return os.path.join(LIB.LIB_DIR, pid, "read_%02d.jpg" % (int(target_pc) % 12))


def page_meta(pid: str) -> dict:
    """这一页的阅读信息（不改库，只读）。"""
    it = LIB.get(pid) or {}
    pg = LIB.group_of(pid)
    key = (it.get("key") or pg.get("group_key") or "").strip()
    kpc = M.parse_key(key) if key else None
    w = h = 0
    try:
        from . import core
        import cv2
        bgr = LIB.decode_bgr(it.get("image") or "")
        if bgr is not None:
            h, w = bgr.shape[:2]
    except Exception:
        pass
    return dict(id=pid, idx=pg.get("idx", 0), total=pg.get("total", 0),
                group=pg.get("group", ""), group_name=pg.get("group_name", ""),
                key=key, key_pc=int(kpc) if kpc is not None else -1,
                w=int(w), h=int(h), prev=pg.get("prev", ""), next=pg.get("next", ""))


def group_meta(pid: str) -> dict:
    m = page_meta(pid)
    gid = m["group"]
    if not gid:
        m["pages"] = [m]
        return m
    ids = LIB.group_pages(gid)
    pages = [page_meta(i) for i in ids]
    m["pages"] = pages
    return m


def render_read(pid: str, target_pc: int) -> Optional[bytes]:
    """渲染/取缓存：这一页在 target 调下的 JPEG 字节。"""
    dst = _read_path(pid, target_pc)
    try:
        if os.path.exists(dst) and os.path.getmtime(dst) >= _cache_mtime(pid):
            return open(dst, "rb").read()
    except Exception:
        pass
    it = LIB.get(pid)
    if not it:
        return None
    meta = page_meta(pid)
    kpc = meta["key_pc"]
    if kpc < 0:
        from . import library as _L
        kpc = M.parse_key(_L.detect_key(_read_txt(pid))) or 0
    cpath = os.path.join(LIB.LIB_DIR, pid, "cache.pkl")
    try:
        c = pickle.load(open(cpath, "rb"))
        page, pairs = c["page"], c["pairs"]
    except Exception:
        txt = _read_txt(pid)
        from .glyphnet import GlyphNet
        _l, pairs, _s, page = MO_merge(it["image"], txt, GlyphNet())
    bgr, _st = EF.render_fixed_image(page, pairs, int(kpc), int(target_pc), mark_missed=True)
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    data = buf.tobytes()
    try:
        with open(dst, "wb") as f:
            f.write(data)
    except Exception:
        pass
    return data


def _read_txt(pid: str) -> str:
    it = LIB.get(pid) or {}
    p = it.get("ocr_txt") or ""
    try:
        return open(p, encoding="utf8").read() if p and os.path.exists(p) else ""
    except Exception:
        return ""


def MO_merge(image: str, txt: str, net):
    from . import merge_octave as MO
    return MO.merge_page(image, txt, net)


def warm(pid: str, target_pc: int, log=print) -> int:
    """把整份谱在 target 调下全部渲染好（阅读前预渲染）。"""
    m = group_meta(pid)
    pages = m.get("pages") or [m]
    n = 0
    for p in pages:
        try:
            if render_read(p["id"], target_pc):
                n += 1
                log("预渲染 %d/%d" % (n, len(pages)))
        except Exception as e:
            log("预渲染失败 %s：%s" % (p["id"][:8], str(e)[:60]))
    return n
