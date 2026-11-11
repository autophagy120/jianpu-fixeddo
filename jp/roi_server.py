# -*- coding: utf-8 -*-
"""ROI 纠错本地服务：转换 → 人工框选 → 一键加强 → 可回退。

设计（按用户要求）：
- **自动回传**：网页 POST 到本机服务，不需要下载文件
- **框选不需要说"正确是什么"**：画个框就表示"这块要加强"，服务端给出**加强方案**
  列表（加强识别 / 加强清除 / 强制插入），点一下就执行并返回新图
- **加强识别**：对框内区域重新跑 OCR（RapidOCR 带坐标），把结果并入音符表
- **加强清除**：把框内墨迹擦白（不做识别）
- **可回退**：每次操作前压栈，点"回退"回到上一步
- **单击** = 选中最近的音符，可改数字/点数后应用
- **双击** = 在该位置强制插入一个音符（数字由输入框决定）

启动：
    python -m jp.roi_server --song BadApple
然后浏览器打开打印出来的 http://127.0.0.1:8765/
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import core
from . import emit_fixed as EF
from . import merge_octave as MO
from . import music as M
from . import xiaomi_ocr as XO

SONGS = {
    "三月雨": (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg",
             r"out\batch\_ocr\三月雨__50b751ebbe319 [原始大小].txt", "bA"),
    "为了你": (r"work\为了你唱下去.jpg", r"work\为了你唱下去.ocr.txt", "G"),
    "BadApple": (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\Bad Apple！！\371057593e3fc10d8bb [原始大小].jpg",
                 r"out\batch\_ocr\Bad Apple！！__371057593e3fc10d8bb [原始大小].txt", "F#"),
}

PORT = 8765

# 显示用调名（`C#` 比 `M.key_name` 的 `#C` 好读；`parse_key` 两种都认）
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class State:
    """服务端状态：音符表 + 额外擦除框 + 回退栈。"""

    def __init__(self, orig: str, ocr_path: str, key_name: str, title: str = "",
                 cache_key: str = "", strength: int = 1):
        from .glyphnet import GlyphNet
        self.orig = orig
        self.ocr_path = ocr_path
        self.song = title or os.path.splitext(os.path.basename(orig))[0]
        self.key_pc = M.parse_key(key_name)
        if self.key_pc is None:
            self.key_pc = 0
        # ---- 打开缓存：第一次慢（几何 + OCR 对齐），之后直接读 pickle
        import jp.library as LIB
        self.cache_key = cache_key or ""
        self.strength = int(strength if strength is not None else 1)
        self.ocr_txt = ''
        _txt = ""
        try:
            _txt = open(ocr_path, encoding="utf8").read()
        except Exception:
            pass
        self.ocr_txt = _txt or ""
        _osig = LIB.ocr_sig(ocr_path)
        cached = LIB.load_cache(self.cache_key, orig, _osig) if self.cache_key else None
        if cached is not None and int(cached.get("strength", -1)) != self.strength:
            cached = None
        if cached:
            self.page = cached["page"]
            self.pairs: List[Tuple[core.Note, XO.OcrNote]] = list(cached["pairs"])
            self.net = None
        else:
            txt = MO.sanitize_ocr(_txt or open(ocr_path, encoding="utf8").read())
            net = GlyphNet()
            _lines, pairs, _st, page = MO.merge_page(orig, txt, net, strength=self.strength)
            self.page = page
            self.net = net
            self.pairs = list(pairs)
            if self.cache_key:
                LIB.save_cache(self.cache_key,
                               dict(page=page, pairs=self.pairs, src_sig=LIB.src_sig(orig),
                                    ocr_sig=_osig, strength=int(self.strength),
                                    built=time.strftime("%Y-%m-%d %H:%M:%S")))
        self.med_h = float(np.median([max(1.0, s.box[3] - s.box[1]) for s, _o in self.pairs])) or 20.0
        self.erase_boxes: List[Tuple[int, int, int, int]] = []
        self.history: List[dict] = []
        self._bgr = None
        self.last_bbox = None
        self.last_refresh_lazy = False
        self.uncovered: List[core.Comp] = []
        # 中途转调（`转1=G`）：锚点是**图上的点**，不是行号 —— 转调可能从一行的
        # 半路开始。语义：阅读顺序在该点之后的音用新调，之前的用旧调。
        self.key_changes: List[dict] = []
        self.target_pc = self.key_pc
        if cached:                                  # 上次手动改过的设置也一并恢复
            self.key_changes = [dict(c) for c in (cached.get("key_changes") or [])]
            if cached.get("target_pc") is not None:
                self.target_pc = int(cached["target_pc"])
            self.erase_boxes = [tuple(b) for b in (cached.get("erase_boxes") or [])]
        self.lock = threading.Lock()
        self.refresh_uncovered()
        self.png = self._render()
        try:
            _g = np.stack([self.page.gray] * 3, axis=-1)[:, :, ::-1]
            _buf = io.BytesIO()
            from PIL import Image as _Im
            _Im.fromarray(_g).save(_buf, format="PNG", compress_level=1)
            self.orig_png = _buf.getvalue()
        except Exception:
            self.orig_png = b""

    def save_cache(self) -> bool:
        """把当前状态（含手工修改）写回库缓存，下次打开是秒开。"""
        if not self.cache_key:
            return False
        import jp.library as LIB
        _osig = ""
        try:
            import jp.library as _L2
            _osig = _L2.ocr_sig(self.ocr_path)
        except Exception:
            pass
        return LIB.save_cache(self.cache_key,
                              dict(page=self.page, pairs=self.pairs,
                                   src_sig=LIB.src_sig(self.orig),
                                   ocr_sig=_osig,
                                   key_changes=self.key_changes,
                                   target_pc=int(self.target_pc),
                                   erase_boxes=list(self.erase_boxes),
                                   saved=time.strftime("%Y-%m-%d %H:%M:%S")))

    # ---------------- 渲染
    def _render(self) -> bytes:
        bgr, _st = EF.render_fixed_image(self.page, self.pairs, self.key_pc, self.target_pc,
                                         mark_missed=True, keys=self.keys_for_pairs())
        for (x0, y0, x1, y1) in self.erase_boxes:
            a, b = max(0, int(y0)), min(bgr.shape[0], int(y1))
            c, d = max(0, int(x0)), min(bgr.shape[1], int(x1))
            if b <= a or d <= c:
                continue
            sub = bgr[a:b, c:d]
            is_red = (sub[:,:,2].astype(np.int32) > 110) & (sub[:,:,1].astype(np.int32) < 95) & (sub[:,:,0].astype(np.int32) < 95)
            is_ink = (sub.astype(np.int32).sum(axis=2)) < 480
            mm = is_ink & (~is_red)          # 只擦非红的墨，红字保留
            if mm.any():
                sub[mm] = (255, 255, 255)
        bgr = self._draw_key_markers(bgr)
        self._bgr = bgr                 # 留一份数组：局部刷新直接裁，不重新渲染
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(bgr[:, :, ::-1]).save(buf, format="PNG", optimize=False, compress_level=1)
        return buf.getvalue()

    def _render_only(self):
        """只重算图（~0.09s），不做整页 PNG 编码（~0.20s）。"""
        bgr, _st = EF.render_fixed_image(self.page, self.pairs, self.key_pc, self.target_pc,
                                         mark_missed=True, keys=self.keys_for_pairs())
        for (x0, y0, x1, y1) in self.erase_boxes:
            a, b = max(0, int(y0)), min(bgr.shape[0], int(y1))
            c, d = max(0, int(x0)), min(bgr.shape[1], int(x1))
            if b <= a or d <= c:
                continue
            sub = bgr[a:b, c:d]
            is_red = ((sub[:, :, 2].astype(np.int32) > 110)
                      & (sub[:, :, 1].astype(np.int32) < 95)
                      & (sub[:, :, 0].astype(np.int32) < 95))
            is_ink = (sub.astype(np.int32).sum(axis=2)) < 480
            mm = is_ink & (~is_red)
            if mm.any():
                sub[mm] = (255, 255, 255)
        self._bgr = self._draw_key_markers(bgr)
        self.png = None
        return self._bgr

    # ---------------- 转调
    def _row_centers(self):
        """按 row 归并，算出每行的中心 y 和左右范围。"""
        rows = {}
        for s, _o in self.pairs:
            r = int(s.row)
            cy = (s.box[1] + s.box[3]) / 2.0
            d = rows.setdefault(r, dict(ys=[], x0=1e18, x1=-1e18))
            d["ys"].append(cy)
            d["x0"] = min(d["x0"], s.box[0])
            d["x1"] = max(d["x1"], s.box[2])
        out = []
        for r, d in rows.items():
            ys = sorted(d["ys"])
            out.append(dict(row=r, y=ys[len(ys) // 2], x0=d["x0"], x1=d["x1"],
                            y0=ys[0], y1=ys[-1]))
        out.sort(key=lambda d: d["y"])
        return out

    def _anchor(self, c) -> tuple:
        """把转调锚点吸附到最近一行。

        之前直接拿点击的 y 跟每个音的中心比：点在"行首"但稍微偏上/偏下
        （比如落在减时线或词上），这一行就整个被判成"不在锚点之后"，
        转调就从**下一行**才开始生效——看起来就像"插到后面去了"。
        这里先把 y 吸到最近一行的中心（1.2 个字高内），行首点（x 在该行最左
        音符左边）再往左放宽一点，保证整行都算进新调。
        """
        ax, ay = float(c["x"]), float(c["y"])
        gh = max(8.0, float(self.med_h))
        best = None
        for d in self._row_centers():
            dy = abs(d["y"] - ay)
            if dy <= 1.5 * gh and (best is None or dy < best[0]):
                best = (dy, d)
        if best is None:
            return ax, ay
        d = best[1]
        if ax <= d["x0"] + 0.25 * gh:      # 行首（甚至更左）→ 整行都算
            ax = d["x0"] - 1.0
        return ax, float(d["y"])

    def keys_for_pairs(self) -> Optional[List[int]]:
        """逐音调号。转调锚点按阅读顺序切段，没打 TAG 时返回 None（全谱同调）。"""
        if not self.key_changes:
            return None
        ch = sorted(self.key_changes, key=lambda c: (float(c["y"]), float(c["x"])))
        tol = 0.5 * max(1.0, self.med_h)
        out: List[int] = []
        for s, _o in self.pairs:
            cy = (s.box[1] + s.box[3]) / 2.0
            cx = (s.box[0] + s.box[2]) / 2.0
            cur = int(self.key_pc)
            for c in ch:
                ax, ay = self._anchor(c)
                # 下半行整行算"之后"；同一行（±半个字高）只算该点右侧
                if cy > ay + tol or (cy >= ay - tol and cx >= ax):
                    cur = int(c["key_pc"])
            out.append(cur)
        return out

    def add_key_change(self, x: float, y: float, key_name: str) -> bool:
        pc = M.parse_key(str(key_name))
        if pc is None:
            return False
        self.snapshot()
        self.key_changes.append(dict(x=float(x), y=float(y), key_pc=int(pc)))
        self._rec("keychange_add", x=float(x), y=float(y), key=str(key_name))
        self.refresh(lazy=True)
        return True

    def del_key_change(self, i: int) -> bool:
        if not (0 <= i < len(self.key_changes)):
            return False
        self.snapshot()
        self.key_changes.pop(i)
        self.refresh(lazy=True)
        return True

    def set_target(self, key_name: str) -> bool:
        pc = M.parse_key(str(key_name))
        if pc is None:
            return False
        self.snapshot()
        self._rec("target", key_old=str(KEY_NAMES[int(self.target_pc) % 12]),
                  key_new=str(key_name))
        self.target_pc = int(pc)
        self.refresh(lazy=True)
        return True

    def state(self) -> dict:
        return dict(strength=int(self.strength), notes=self.notes(),
                    source_key=KEY_NAMES[int(self.key_pc) % 12],
                    target_key=KEY_NAMES[int(self.target_pc) % 12],
                    key_changes=[dict(x=float(c["x"]), y=float(c["y"]),
                                      key=KEY_NAMES[int(c["key_pc"]) % 12])
                                 for c in self.key_changes])

    def _snap_marker_x(self, ax: float, ay: float) -> float:
        """把标记柱挪到"分界音前一个音"和"分界音"之间的空档里，别压在数字上。

        切段语义不变（仍按点击位置），只是柱子画在段落边界处更好看。
        """
        gh = max(8.0, float(self.med_h))
        tol = 0.5 * gh
        row = []
        for s, _o in self.pairs:
            cy = (s.box[1] + s.box[3]) / 2.0
            if abs(cy - ay) <= tol:
                row.append(s.box)
        if not row:
            return ax
        row.sort(key=lambda b: b[0])
        nxt = None
        for b in row:
            if (b[0] + b[2]) / 2.0 >= ax:
                nxt = b
                break
        if nxt is None:                       # 点在整行最右之后：贴着最后一个音右边
            return min(row[-1][2] + 0.18 * gh, ax)
        prev = [b for b in row if b[2] <= nxt[0] + 1]
        if not prev:                          # 行首
            return max(0.0, nxt[0] - 0.35 * gh)
        return (prev[-1][2] + nxt[0]) / 2.0

    def _draw_key_markers(self, bgr: np.ndarray) -> np.ndarray:
        """图上标出转调点：竖直紫色虚线 + `1=G` 标签。"""
        if not self.key_changes:
            return bgr
        from PIL import Image as _I, ImageDraw as _D
        h, w = bgr.shape[:2]
        gh = max(8.0, float(self.med_h))
        col = (150, 40, 200)                       # BGR 紫
        th = max(2, int(round(gh * 0.10)))
        im = _I.fromarray(bgr[:, :, ::-1])
        dr = _D.Draw(im)
        dash = max(6, int(round(gh * 0.40)))
        gap = max(4, int(round(gh * 0.26)))
        font = EF._font(max(11, int(round(gh * 0.62))))
        for c in self.key_changes:
            _ax, _ay = self._anchor(c)
            ax = int(round(self._snap_marker_x(_ax, _ay)))
            ay = int(round(_ay))
            y0 = max(0, int(ay - 1.7 * gh))
            y1 = min(h - 1, int(ay + 1.7 * gh))
            y = y0
            while y < y1:
                dr.rectangle([ax - th // 2, y, ax - th // 2 + th - 1, min(y1, y + dash)],
                             fill=col)
                y += dash + gap
            txt = "1=" + KEY_NAMES[int(c["key_pc"]) % 12]
            bb = font.getbbox(txt)
            tw, thh = bb[2] - bb[0], bb[3] - bb[1]
            lx = min(max(0, ax + int(0.10 * gh)), max(0, w - tw - 6))
            ly = max(0, y0 - thh - int(0.35 * gh))
            dr.rectangle([lx - 3, ly - 2, lx + tw + 3, ly + thh + 3],
                         fill=(255, 255, 255), outline=col, width=max(1, th // 2))
            dr.text((lx, ly), txt, font=font, fill=col)
        return np.asarray(im)[:, :, ::-1].copy()

    def snapshot(self):
        try:
            import jp.edits as _ED
            self._snap_seq = _ED.cur_seq()
        except Exception:
            self._snap_seq = 0
        """回退栈要连数据一起存 —— 只存 png 的话，回退后一刷新变化又回来了。"""
        self.history.append(dict(png=self.png,
                                 pairs=copy.deepcopy(self.pairs),
                                 erase=list(self.erase_boxes),
                                 uncovered=list(self.uncovered),
                                 kc=copy.deepcopy(self.key_changes),
                                 target=int(self.target_pc),
                                 seq=int(getattr(self, "_snap_seq", 0))))
        if len(self.history) > 40:
            self.history.pop(0)

    def _glyph_slots(self):
        """没有 OCR 文本的页（人工导入）：把像数字的墨块都当成待填槽位。

        这样打开后满屏橙框，点一个填一个；比"空白页 + 自己找位置"顺手得多。
        """
        comps = list(getattr(self.page, "comps", []) or [])
        if not comps:
            return []
        hs = sorted(float(c.h) for c in comps)
        # 用版面字高（page.scale）当基准：全部连通域的中位高会被小点/横线带偏
        med = float(getattr(self.page, "scale", 0) or 0)
        if med < 6:
            med = hs[len(hs) // 2] if hs else 20.0
        covered = [(float(s2.box[0]), float(s2.box[1]), float(s2.box[2]), float(s2.box[3]))
                   for s2, _o in self.pairs]
        out = []
        for c in comps:
            h = float(c.h) or 1.0
            w = float(c.w) or 1.0
            if h < 0.62 * med or h > 1.45 * med:
                continue
            if not (0.35 <= w / h <= 1.25):
                continue
            cx, cy = c.x + c.w / 2.0, c.y + c.h / 2.0
            if any(b[0] - 2 <= cx <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2 for b in covered):
                continue
            out.append(c)
        # 再用数字分类器过一遍：歌词/标点也被切成同尺寸的块，
        # 不过一遍会在歌词上画满橙框（实测 573 个，其中一半不是音符）
        try:
            from . import core as _core
            if self.net is None:
                from .glyphnet import GlyphNet
                self.net = GlyphNet()
            cand = [c for c in out if c.w >= 4 and c.h >= 4]
            patches = [_core.patch_of(self.page.clean, c) for c in cand]
            if patches:
                labs, _cf = self.net.digit_with_reject(patches, min_prob=0.45)
                out = [c for c, lab in zip(cand, labs) if lab]
        except Exception:
            pass
        out.sort(key=lambda c: (round(c.y / max(4.0, 0.8 * med)), c.x))
        return out

    def refresh_uncovered(self):
        try:
            if not (self.ocr_txt or "").strip():
                self.uncovered = self._glyph_slots()
            else:
                self.uncovered = EF.find_uncovered(self.page, self.pairs)
        except Exception:
            self.uncovered = []

    def _rec(self, kind: str, **kw):
        """记一条修正事件（自学习用）。失败绝不影响主流程。"""
        try:
            import jp.edits as ED
            box = kw.pop("box", None)
            kw.setdefault("ctx", self._ctx(box))
            if box is not None:
                kw["box"] = [float(v) for v in box]
            return ED.log_event(self.cache_key or self.song, kind, **kw)
        except Exception:
            return None

    def _rec_sample(self, box, label: str, pred: str = "", kind: str = "", **extra):
        """按框从 page.clean 抠字形 → 存成训练样本（和训练同域）。"""
        try:
            import jp.edits as ED
            patch = ED.patch_from_clean(self.page.clean, box)
            if label != "blank" and not ED.has_ink(patch):
                return None
            return ED.add_sample(patch, label, self.cache_key or self.song, box,
                                 pred=pred, kind=kind, extra=extra or None,
                                 ctx=self._ctx(box))
        except Exception:
            return None

    def _ctx(self, box=None) -> dict:
        """谱面信息 + 这个音周围的行/邻音坐标（坐标系和训练一致：原图像素）。"""
        H, W = self.page.gray.shape[:2]
        d = dict(w=int(W), h=int(H), scale=round(float(getattr(self.page, "scale", 0) or 0), 4),
                 med_h=round(float(self.med_h), 2), key=str(KEY_NAMES[int(self.key_pc) % 12]),
                 target=str(KEY_NAMES[int(self.target_pc) % 12]),
                 strength=int(self.strength), pipe=int(getattr(__import__("jp.library", fromlist=["x"]), "PIPE_VER", 0)),
                 img=os.path.basename(self.orig))
        if box is None:
            return d
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
        d["box"] = [round(float(v), 1) for v in box]
        d["ph"] = round(float(box[3]) - float(box[1]), 1)
        d["pw"] = round(float(box[2]) - float(box[0]), 1)
        # 同一行里左右最近的音（学习"错位"最需要的信号）
        row = self._row_for(cx, cy)
        d["row"] = int(row)
        same = [(k, s2) for k, (s2, _o2) in enumerate(self.pairs)
                if int(s2.row) == int(row)]
        left = [s2 for _k, s2 in same if (s2.box[0] + s2.box[2]) / 2.0 < cx]
        right = [s2 for _k, s2 in same if (s2.box[0] + s2.box[2]) / 2.0 > cx]
        if left:
            l = max(left, key=lambda s2: s2.box[2])
            d["left"] = [round(float(l.box[0]), 1), round(float(l.box[2]), 1),
                         int(l.box[2] - (l.box[2] - l.box[0]))]
            d["dx_left"] = round(cx - (l.box[0] + l.box[2]) / 2.0, 1)
        if right:
            r = min(right, key=lambda s2: s2.box[0])
            d["right"] = [round(float(r.box[0]), 1), round(float(r.box[2]), 1), 0]
            d["dx_right"] = round((r.box[0] + r.box[2]) / 2.0 - cx, 1)
        d["row_n"] = len(same)
        base = [float(s2.box[3]) for _k, s2 in same]
        if base:
            base.sort()
            d["row_base"] = round(base[len(base) // 2], 1)
        return d

    def _note_bbox(self, boxes, pad_gh: float = 2.2):
        """若干字形框外扩 → 一个矩形（像素，clamp 到图内）。"""
        if not boxes:
            return None
        gh = max(8.0, float(self.med_h)) * pad_gh
        x0 = min(float(b[0]) for b in boxes) - gh
        y0 = min(float(b[1]) for b in boxes) - gh
        x1 = max(float(b[2]) for b in boxes) + gh
        y1 = max(float(b[3]) for b in boxes) + gh
        h, w = self.page.gray.shape[:2]
        return [int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))]

    def patch_png(self, x: int, y: int, w: int, h: int):
        """裁出当前渲染图的一块，编码成小 PNG（20-30KB，1-2ms）。"""
        if self._bgr is None:
            return None
        H, W = self._bgr.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        crop = self._bgr[y0:y1, x0:x1]
        from PIL import Image as _Im
        buf = io.BytesIO()
        _Im.fromarray(crop[:, :, ::-1]).save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    def refresh(self, lazy: bool = False):
        """lazy=True：只重算数组，不编码整页 PNG（改音符走这条，快约 3 倍）。"""
        self.refresh_uncovered()
        if lazy:
            self._render_only()
        else:
            self.png = self._render()
        self.last_refresh_lazy = bool(lazy)
        self._log("render pairs=%d uncovered=%d keychanges=%d target=%s%s"
                  % (len(self.pairs), len(self.uncovered), len(self.key_changes),
                     KEY_NAMES[int(self.target_pc) % 12], " (lazy)" if lazy else ""))

    # ---------------- 查询
    def notes(self):
        out = []
        for i, (s, o) in enumerate(self.pairs):
            out.append(dict(id=i,
                            x=float(s.box[0]), y=float(s.box[1]),
                            w=float(s.box[2] - s.box[0]), h=float(s.box[3] - s.box[1]),
                            digit=str(o.digit), oct=int(o.octave),
                            row=int(s.row),
                            sc=str(getattr(s, "size_class", "main"))))
        # 橙框（漏识别）用负 id 排在后面，前端点中后走 /assign 回填
        for j, c in enumerate(self.uncovered):
            out.append(dict(id=-(j + 1), u=1,
                            x=float(c.x), y=float(c.y), w=float(c.w), h=float(c.h),
                            digit="?", oct=0, row=-1))
        return out

    def _row_for(self, x: float, y: float) -> int:
        best, bd = -1, 1e18
        for s, _o in self.pairs:
            cy = (s.box[1] + s.box[3]) / 2.0
            d = abs(cy - y)
            if d < bd:
                bd, best = d, int(s.row)
        return best

    def _resort(self):
        self.pairs.sort(key=lambda p: (int(p[0].row), p[0].box[1], p[0].box[0]))

    def nearest(self, x: float, y: float) -> Optional[int]:
        best, bd = None, 1e18
        for i, (s, _o) in enumerate(self.pairs):
            x0, y0, x1, y1 = s.box
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d = (cx - x) ** 2 + (cy - y) ** 2
            if x0 - s.box[2] + x0 <= x <= x1 + (x1 - x0) and y0 - 2 * (y1 - y0) <= y <= y1 + 2 * (y1 - y0):
                d *= 0.15
            if d < bd:
                bd, best = d, i
        return best

    def in_roi(self, roi) -> List[int]:
        x, y, w, h = roi
        out = []
        for i, (s, _o) in enumerate(self.pairs):
            x0, y0, x1, y1 = s.box
            if x1 >= x and x0 <= x + w and y1 >= y and y0 <= y + h:
                out.append(i)
        return out

    # ---------------- 操作
    def plan_for(self, roi) -> List[str]:
        """给出可选的加强方案（不需要用户说正确内容）。"""
        ids = self.in_roi(roi)
        plans = ["加强识别（对框内重新 OCR）", "加强清除（把框内墨迹擦白）",
                 "删除框内所有音符", "强制插入音符", "取消"]
        if ids:
            plans.insert(0, f"框内有 {len(ids)} 个已识别音符（可单击修改）")
        else:
            plans.insert(0, "框内没有已识别音符 → 建议『加强识别』或『强制插入』")
        return plans

    def strengthen(self, mode: str, roi):
        x, y, w, h = [int(v) for v in roi]
        self.snapshot()
        info = dict(found=0, added=0, skipped=0)
        if mode == "erase":
            self.erase_boxes.append((x, y, x + w, y + h))
        elif mode == "ocr":
            info = self._ocr_roi(x, y, w, h)
        self._log("strengthen mode=%s roi=(%d,%d,%d,%d) found=%d added=%d skipped=%d pairs=%d"
                  % (mode, x, y, w, h, info.get("found", 0), info.get("added", 0),
                     info.get("skipped", 0), len(self.pairs)))
        self.refresh(lazy=True)
        return info

    def _log(self, msg: str):
        """把每次操作写进 out\\roi_server.log —— 出问题能回看而不是靠猜。"""
        try:
            os.makedirs("out", exist_ok=True)
            with open(os.path.join("out", "roi_server.log"), "a", encoding="utf8") as f:
                f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
        except Exception:
            pass

    def _ocr_roi(self, x, y, w, h) -> dict:
        """对框内区域重新 OCR，**只增不删**地把新数字并进音符表。

        以前是"先删框内原有音符、再把 OCR 结果补回来"，一旦 OCR 没读回数字
        （比如框里主要是歌词/橙框、或者识别合并了字形），那些音就凭空消失，
        用户看到的就是"框第二个框的时候第一个框和第二个框里的音全没了"。
        现在改成：原有音符一个都不动，只补"附近还没有音符"的数字。
        改错音仍然走单击修改，不会再有静默删除。
        """
        from . import ocr_boxes as OB
        info = dict(found=0, added=0, skipped=0)
        h_img, w_img = self.page.gray.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_img, x + w), min(h_img, y + h)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return info
        crop = self.page.gray[y0:y1, x0:x1]
        try:
            boxes, _s = OB.read_boxes(crop)
            chars = OB.chars_from_boxes(boxes, digits_only=False)
        except Exception:
            return info
        if not chars:
            return info
        info["found"] = len(chars)
        med = self.med_h
        tol = max(4.0, 0.65 * med)
        centers = [((s.box[0] + s.box[2]) / 2.0, (s.box[1] + s.box[3]) / 2.0)
                   for s, _o in self.pairs]
        for c in chars:
            if not c.ch.isdigit():
                continue
            cx, cy = x0 + c.cx, y0 + c.cy
            # 这个位置已经有音符 → 不重复添加（也不删除原来的）
            if any(abs(cx - px) <= tol and abs(cy - py) <= tol for px, py in centers):
                info["skipped"] += 1
                continue
            gw = max(6.0, c.w * 0.9)
            gh = max(8.0, c.h * 0.9)
            nb = core.Note(digit=str(c.ch), x=cx,
                           box=(cx - gw / 2, cy - gh / 2, cx + gw / 2, cy + gh / 2),
                           row=self._row_for(cx, cy), conf=0.5)
            self.pairs.append((nb, XO.OcrNote(digit=str(c.ch), octave=0)))
            centers.append((cx, cy))
            info["added"] += 1
        if info["added"]:
            self._resort()
        return info

    def del_note(self, idx: int) -> bool:
        """删掉一个音：加强识别偶尔会多认，必须能单独撤掉，不然只能整体回退。"""
        if not (0 <= idx < len(self.pairs)):
            return False
        self.snapshot()
        self.pairs.pop(int(idx))
        self._resort()
        self._log("del id=%d pairs=%d" % (idx, len(self.pairs)))
        self.refresh(lazy=True)
        return True

    def edit_note(self, idx: int, digit: str, octv: int):
        if 0 <= idx < len(self.pairs):
            self.snapshot()
            _s, o = self.pairs[idx]
            self.pairs[idx] = (_s, XO.OcrNote(digit=str(digit), octave=max(-2, min(2, int(octv)))))
            self._rec("edit", idx=int(idx), digit_old=str(o.digit), oct_old=int(o.octave),
                      digit_new=str(digit), oct_new=int(octv),
                      box=[float(v) for v in _s.box])
            self._rec_sample(_s.box, str(digit), pred=str(o.digit), kind="edit",
                             oct_new=int(octv), oct_old=int(o.octave))
            self.last_bbox = self._note_bbox([_s.box])
            self.refresh(lazy=True)

    def insert_note(self, x: float, y: float, digit: str, octv: int = 0):
        self.snapshot()
        gh = self.med_h
        nb = core.Note(digit=str(digit), x=float(x),
                       box=(x - gh * 0.35, y - gh * 0.5, x + gh * 0.35, y + gh * 0.5),
                       row=self._row_for(x, y), conf=0.3)
        self.pairs.append((nb, XO.OcrNote(digit=str(digit), octave=max(-2, min(2, int(octv))))))
        self._rec("insert", digit=str(digit), oct_new=int(octv),
                  box=[float(v) for v in nb.box], xy=[float(x), float(y)])
        self._rec_sample(nb.box, str(digit), pred="", kind="insert", oct_new=int(octv))
        self._resort()
        self.refresh(lazy=True)

    def assign_slot(self, j: int, digit: str, octv: int) -> Optional[int]:
        """把第 j 个橙框（漏识别墨块）指定成 digit + 八度，变成正式音符。"""
        if not (0 <= j < len(self.uncovered)):
            return None
        self.snapshot()
        c = self.uncovered[j]
        cx, cy = c.x + c.w / 2.0, c.y + c.h / 2.0
        digit = str(digit)
        octv = max(-2, min(2, int(octv)))
        # **先看附近有没有"识别到但位置歪了"的同一个音**：
        # 对齐错位时数字被配到相邻槽位，真正位置成了橙框；直接新增就会变成两个。
        # 所以优先把最近的那一个挪过来（顺便把它"歪了"的位置修正）。
        self._rec("assign", slot=int(j), digit_new=digit, oct_new=octv,
                  box=[float(v) for v in c.box])
        self._rec_sample(c.box, digit, pred="?", kind="assign", oct_new=octv)
        gh = max(8.0, float(self.med_h))
        best, bd = -1, None
        for i, (ss, oo) in enumerate(self.pairs):
            if str(oo.digit) != digit:
                continue
            sy = (ss.box[1] + ss.box[3]) / 2.0
            sx = (ss.box[0] + ss.box[2]) / 2.0
            if abs(sy - cy) > 0.75 * gh:
                continue
            dx = abs(sx - cx)
            if dx <= 2.2 * gh and (bd is None or dx < bd):
                best, bd = i, dx
        if best >= 0:
            ss, _oo = self.pairs[best]
            ss.box = c.box
            ss.x = float(c.x)
            ss.row = self._row_for(cx, cy)
            ss.dot_boxes = []
            ss.acc_box = None
            self.pairs[best] = (ss, XO.OcrNote(digit=digit, octave=octv))
            self._log("assign slot=%d digit=%s -> 把邻近音 #%d 挪过来（dx=%.0f）"
                      % (j, digit, best, bd or 0))
            self._resort()
            self.refresh(lazy=True)
            for i, (s2, _o) in enumerate(self.pairs):
                if s2 is ss:
                    return i
            return best
        s = core.Note(digit=digit, x=float(c.x), box=c.box,
                      row=self._row_for(cx, cy), conf=0.35)
        self.pairs.append((s, XO.OcrNote(digit=digit, octave=octv)))
        self._resort()
        self.refresh(lazy=True)
        for i, (ss, _o) in enumerate(self.pairs):
            if ss is s:
                return i
        return None

    def move_note(self, idx: int, dx: float, dy: float) -> bool:
        """手动微调一个音的位置（对齐错位时用）。

        坐标修正是最难自动学的一类：这里把 **原框** 和 **新框** 一起记下来，
        既能当作"对齐应该偏多少"的回归样本，也能当人工标注的位置。
        """
        if not (0 <= idx < len(self.pairs)):
            return False
        self.snapshot()
        s, o = self.pairs[idx]
        old = tuple(float(v) for v in s.box)
        s.box = (old[0] + dx, old[1] + dy, old[2] + dx, old[3] + dy)
        s.x = float(s.box[0])
        s.row = self._row_for((s.box[0] + s.box[2]) / 2.0, (s.box[1] + s.box[3]) / 2.0)
        self._rec("move", idx=int(idx), digit=str(o.digit), dx=round(float(dx), 1),
                  dy=round(float(dy), 1), box_old=[round(v, 1) for v in old],
                  box_new=[round(v, 1) for v in s.box], box=list(s.box))
        try:
            import jp.edits as ED
            patch = ED.patch_from_clean(self.page.clean, s.box)
            ED.add_sample(patch, str(o.digit), self.cache_key or self.song, s.box,
                          pred=str(o.digit), kind="move",
                          extra={"box_old": [round(v, 1) for v in old],
                                 "dx": round(float(dx), 1), "dy": round(float(dy), 1)},
                          ctx=self._ctx(s.box))
        except Exception:
            pass
        self._resort()
        self.last_bbox = self._note_bbox([old, s.box], pad_gh=3.0)
        self.refresh(lazy=True)
        return True

    def set_size(self, ids, cls: str) -> int:
        """把若干音标成主字号 / 辅助字号，返回改动个数。"""
        cls = "aux" if str(cls) == "aux" else "main"
        n = 0
        todo = [i for i in ids if 0 <= int(i) < len(self.pairs)]
        if not todo:
            return 0
        self.snapshot()
        _boxes = []
        for i in todo:
            s, _o = self.pairs[int(i)]
            try:
                s.size_class = cls
            except Exception:
                continue
            _boxes.append([float(v) for v in s.box])
            n += 1
        self._rec("size", cls=str(cls), n=int(n), boxes=_boxes[:40])
        self.last_bbox = self._note_bbox(_boxes)
        self.refresh(lazy=True)
        return n

    def set_size_row(self, row: int, cls: str) -> int:
        """整行统一字号（用于"本行全是小字"或"整行恢复主字号"）。"""
        ids = [i for i, (s, _o) in enumerate(self.pairs) if int(s.row) == int(row)]
        return self.set_size(ids, cls)

    def _rec_blank(self, roi):
        """加强清除擦掉的一块 → blank 负样本（训练"这里不是音符"）。"""
        try:
            import jp.edits as ED
            x, y, w, h = [float(v) for v in roi]
            patch = ED.patch_from_clean(self.page.clean, (x, y, x + w, y + h))
            ED.add_sample(patch, "blank", self.cache_key or self.song,
                          (x, y, x + w, y + h), pred="", kind="erase")
        except Exception:
            pass

    def set_strength(self, level: int) -> bool:
        """换识别强度：用同一份 OCR 文本重跑几何+对齐（约 10-30 秒）。"""
        level = max(0, min(2, int(level)))
        if level == self.strength:
            return True
        from .glyphnet import GlyphNet
        txt = ""
        try:
            txt = open(self.ocr_path, encoding="utf8").read()
        except Exception:
            return False
        txt = MO.sanitize_ocr(txt)
        net = GlyphNet()
        self.snapshot()
        _l, pairs, _st, page = MO.merge_page(self.orig, txt, net, strength=level)
        self.page = page
        self.net = net
        self.pairs = list(pairs)
        self.med_h = float(np.median([max(1.0, s.box[3] - s.box[1]) for s, _o in self.pairs])) or 20.0
        self.strength = level
        self._log("set_strength=%d pairs=%d" % (level, len(self.pairs)))
        self.refresh(lazy=True)
        if self.cache_key:
            import jp.library as _L
            _L.update(self.cache_key, strength=level)
        return True

    def del_notes(self, ids) -> int:
        """批量删除音符（误识别一次清一批）。"""
        want = set()
        for i in ids:
            try:
                want.add(int(i))
            except Exception:
                pass
        drop = {i for i in want if 0 <= i < len(self.pairs)}
        if not drop:
            return 0
        self.snapshot()
        for k in sorted(drop):
            _s0, _o0 = self.pairs[k]
            self._rec("delete", idx=int(k), digit_old=str(_o0.digit),
                      oct_old=int(_o0.octave), box=[float(v) for v in _s0.box])
            self._rec_sample(_s0.box, "reject", pred=str(_o0.digit), kind="delete")
        keep = [(s, o) for k, (s, o) in enumerate(self.pairs) if k not in drop]
        n = len(self.pairs) - len(keep)
        self.pairs = keep
        self._log("del_notes 删除 %d 个" % n)
        self.refresh(lazy=True)
        return n

    def clear_notes_in_roi(self, roi) -> int:
        """把框内的所有已识别音符删掉（整块误识别时一键清）。"""
        return self.del_notes(self.in_roi(roi))

    def undo(self):
        if not self.history:
            return False
        h = self.history.pop()
        try:
            import jp.edits as _ED
            self._rec("undo", seq_from=int(h.get("seq") or 0), seq_to=int(_ED.cur_seq()))
        except Exception:
            pass
        self.png = h["png"]
        self.pairs = h["pairs"]
        self.erase_boxes = h["erase"]
        self.uncovered = h["uncovered"]
        self.key_changes = h["kc"]
        self.target_pc = h["target"]
        return True


PAGE = '<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>简谱纠错 — __SONG__</title>\n<style>\n:root{--bg:#0f1216;--panel:#171b21;--card:#1e242c;--line:#2b333d;--fg:#e6e9ee;--dim:#8b95a3;--acc:#4da3ff;--warn:#ffd166}\n*{box-sizing:border-box}\nhtml,body{margin:0;height:100%}\nbody{background:var(--bg);color:var(--fg);font:14px/1.55 "Microsoft YaHei","Segoe UI",sans-serif;overflow:hidden}\n#app{display:flex;height:100vh}\n#view{flex:1;display:flex;flex-direction:column;min-width:0}\n#tools{display:flex;align-items:center;gap:8px;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}\n#tools .t{color:var(--dim);font-size:12px}\n.sp{flex:1}\n#wrap{flex:1;overflow:auto;background:#0a0c0f;padding:12px}\n#cv{display:block;background:#fff;box-shadow:0 6px 24px rgba(0,0,0,.55);border-radius:4px;cursor:crosshair}\n#side{width:352px;background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column}\n#side header{padding:12px 14px;border-bottom:1px solid var(--line)}\n#side header h1{margin:0;font-size:16px}\n#side header p{margin:4px 0 0;color:var(--dim);font-size:12px}\n#side .body{flex:1;overflow:auto;padding:12px}\n.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:11px;margin-bottom:10px}\n.card h2{margin:0 0 8px;font-size:12px;color:var(--dim);letter-spacing:.4px;font-weight:600}\n.card.hot{border-color:var(--warn);box-shadow:0 0 0 1px rgba(255,209,102,.35) inset}\n.plan{display:flex;align-items:center;gap:9px;background:#232b34;border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:9px 10px;margin-bottom:7px;cursor:pointer;transition:.12s}\n.plan:hover{background:#2b3643;transform:translateX(2px)}\n.plan .ic{width:20px;text-align:center}\n.plan .tx{flex:1;font-size:13px}\nbutton{background:#26313d;color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:8px 12px;font-size:13px;cursor:pointer;transition:.12s}\nbutton:hover{background:#31404f;border-color:var(--acc)}\nbutton.on{background:#1d4e79;border-color:var(--acc);color:#fff}\nbutton.oc{min-width:56px}\n.row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}\n.g{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}\n.t2{color:var(--dim);font-size:12px;margin:2px 0 6px}\nbutton:active{transform:scale(.94);background:#3a4a5c;border-color:var(--acc)}\nbutton[disabled]{opacity:.5;cursor:progress}\n#busy{position:fixed;inset:0;background:rgba(10,12,15,.55);display:none;align-items:center;justify-content:center;z-index:999}\n#busy.on{display:flex}\n#busy .b{background:#1e242c;border:1px solid var(--acc);border-radius:10px;padding:16px 26px;font-size:15px;display:flex;align-items:center;gap:12px;box-shadow:0 12px 40px rgba(0,0,0,.6)}\n#spin{width:18px;height:18px;border:3px solid #2b3a4a;border-top-color:var(--acc);border-radius:50%;animation:sp .8s linear infinite}\n@keyframes sp{to{transform:rotate(360deg)}}\n#kbdTip{color:var(--dim);font-size:12px;margin-left:8px}\n.kbd{background:#232b34;border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px monospace;color:#c7d0da}\n#log{color:var(--dim);font-size:12px;white-space:pre-wrap;max-height:150px;overflow:auto}\n#selInfo{font-size:13px}#selInfo b{color:var(--warn)}\n#stat{font-size:12px;color:var(--dim)}\n#stat b{color:var(--warn)}\n</style></head><body>\n<div id="app">\n  <div id="view">\n    <div id="tools">\n      <button onclick="location.href=\'/lib\'">← 乐谱库</button>\n      <span class="t">__SONG__</span>\n      <button onclick="zoom(-1)">－</button><span class="t" id="zl">—</span><button onclick="zoom(1)">＋</button>\n      <button onclick="fitWidth()">适应宽度</button>\n      <span class="sp"></span>\n      <span id="stat"></span>\n      <button onclick="jumpNext()">跳到下一个橙框</button>\n      <span class="t">识别强度</span>\n      <select id="stg" onchange="setStrength(this.value)" title="严格=只信 OCR 文本（橙框多）；标准；加强=再用几何分类器补字（覆盖率最高）">\n        <option value="0">严格</option><option value="1">标准</option><option value="2">加强</option>\n      </select>\n      <button onclick="saveLib()" title="把当前识别结果和修改写进库缓存，下次打开秒开">保存到库</button>\n      <button id="kcBtn" onclick="toggleKc()">标记转调</button>\n      <span class="t">目标调</span><select id="tgt" onchange="setTarget(this.value)"></select>\n      <button onclick="undo()">↶ 回退</button>\n      <button onclick="reload()">重新渲染</button>\n      <span class="t" id="buildTag" title="页面构建时间；对不上就是浏览器缓存">build __BUILD__</span>\n      <span id="kbdTip">按住 Ctrl 看原图</span>\n    </div>\n    <div id="wrap"><canvas id="cv"></canvas></div>\n  </div>\n  <div id="side">\n    <header><h1>简谱纠错</h1><p>点橙框回填 · 单击改音 · 双击插入 · 拖拽加强</p></header>\n    <div class="body">\n      <div class="card" id="selCard" style="display:none">\n        <h2><span id="selTitle">选中音符</span>\u3000<span id="selInfo"></span></h2>\n        <div class="t2" id="selDigitTip">数字（点一下即改）</div><div class="g" id="selDigits"></div>\n        <div class="t2">八度（点一下即改）</div><div class="g" id="selOcts"></div>\n        <div class="t2">字号</div><div class="g" id="selSizes"></div>\n        <div class="row"><button onclick="delNote()">删除这个音</button>\n          <button class="p" onclick="delSelected()">删除选中(<span id="selCnt">1</span>)</button>\n          <button onclick="clearSel()">取消选中</button></div>\n        <div class="t2">Ctrl + 点击 = 多选（一次删一批） · <b>Alt + 拖动 = 微调音符位置</b></div>\n      </div>\n      <div class="card" id="insCard" style="display:none">\n        <h2>插入音符\u3000<span id="insInfo"></span></h2>\n        <div class="t2">点数字 = 直接插入（八度按当前的）</div><div class="g" id="insDigits"></div>\n        <div class="t2">八度点（点一下 = 带点插入）</div><div class="g" id="insOcts"></div>\n        <div class="t2">键盘 0-7 直接插 · Esc 取消</div>\n        <div class="row"><button onclick="hideIns()">取消</button></div>\n      </div>\n      <div class="card" id="roiCard" style="display:none">\n        <h2>框选区域\u3000<span id="roiCnt"></span></h2>\n        <div id="plans"></div>\n        <div class="row"><button onclick="clearRoi()">清除框选</button></div>\n      </div>\n      <div class="card" id="kcCard" style="display:none">\n        <h2>转调点\u3000<span id="kcInfo"></span></h2>\n        <div class="t2">从这个位置之后的音起 <b>1=?</b></div>\n        <div class="g" id="kcKeys"></div>\n        <div class="row"><button onclick="hideKc()">取消</button></div>\n      </div>\n      <div class="card" id="kcListCard" style="display:none">\n        <h2>已标记的转调点</h2>\n        <div id="kcList"></div>\n      </div>\n      <div class="card">\n        <h2>怎么用</h2>\n        <div style="font-size:12.5px;color:var(--dim);line-height:2">\n          <div><span class="kbd">点橙框</span> 回填漏识别的音（选数字，再点八度）</div>\n          <div><span class="kbd">标记转调</span> 点谱面上任意一点（可从半行起）→ 选 1=?</div>\n          <div><span class="kbd">单击</span> 选中音符 → 右侧点数字、点八度即改</div>\n          <div><span class="kbd">双击空白</span> 插入音符（数字+八度按钮）</div>\n          <div><span class="kbd">拖拽</span> 框选 → 加强识别 / 加强清除</div>\n          <div><span class="kbd">回退</span> 撤销上一步</div>\n        </div>\n      </div>\n      <div id="busy"><div class="b"><div id="spin"></div><span id="busyTx">处理中…</span></div></div>\n      <div class="card" style="padding:8px 10px">\n        <h2 style="display:flex;justify-content:space-between">日志 <span id="logTag" style="color:#5a6673;font-weight:400">v__BUILD__</span></h2>\n        <div id="log" style="min-height:86px;max-height:170px;overflow:auto;background:#0d1116;border:1px solid #232b34;border-radius:6px;padding:6px 8px">（操作记录会出现在这里）</div>\n      </div>\n    </div>\n  </div>\n</div>\n<script>\nvar cv=document.getElementById(\'cv\'), ctx=cv.getContext(\'2d\'), wrap=document.getElementById(\'wrap\');\nvar OCTS=[[-2,\'低2\'],[-1,\'低1\'],[0,\'中\'],[1,\'高1\'],[2,\'高2\']];\nvar KCKEYS=[\'C\',\'C#\',\'D\',\'D#\',\'E\',\'F\',\'F#\',\'G\',\'G#\',\'A\',\'A#\',\'B\'];\nvar kcMode=false, kcAt=null, ST={notes:[],key_changes:[],source_key:\'\',target_key:\'\'};\nvar img=new Image(), imgOrig=new Image(), showOrig=false, NOTES=[], scale=1, zoomPct=100;\nvar base=document.createElement(\'canvas\'), bctx=base.getContext(\'2d\'), baseReady=false;\nvar moveDrag=null;   // Alt+拖动 = 微调音符坐标\nvar selId=null, selIds=[], selData=null, selDigit=\'6\', roi=null, plans=[], drag=null, insAt=null, insDigit=\'6\', insOct=0, clickTimer=null;\n\nfunction log(s){var e=document.getElementById(\'log\');e.textContent=(new Date().toLocaleTimeString()+\'  \'+s+\'\\n\'+e.textContent).slice(0,2000);}\nvar pending=0;\nfunction busy(on,txt){\n  var el=document.getElementById(\'busy\');\n  document.getElementById(\'busyTx\').textContent=txt||\'处理中…\';\n  el.className=on?\'on\':\'\';\n}\nfunction hold(txt){ pending++; busy(true,txt||\'处理中…\'); }\nfunction release(){\n  pending=Math.max(0,pending-1);\n  if(!pending){ busy(false); document.querySelectorAll(\'button\').forEach(function(x){x.disabled=false;}); }\n}\nfunction api(p,b){\n  hold(\'处理中…\');\n  document.querySelectorAll(\'button\').forEach(function(x){x.disabled=true;});\n  return fetch(p,{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},body:JSON.stringify(b||{})})\n    .then(function(r){return r.json();})\n    .finally(release);\n}\n/* 一次改动的完整收尾：取新数据 → 等新图真正画到 canvas 上，整段都占着转圈 */\nfunction applyRender(){\n  hold(\'重新绘制…\');\n  return loadNotes().then(reload).finally(release);\n}\nfunction boxOf(id){\n  for(var i=0;i<NOTES.length;i++){var n=NOTES[i];\n    if(n.id===id) return [n.x,n.y,n.w,n.h];}\n  return null;\n}\nfunction patchRegion(x,y,w,h){\n  var pad=90;\n  var X=Math.max(0,Math.floor(x-pad)), Y=Math.max(0,Math.floor(y-pad));\n  var W=Math.ceil(w+pad*2), H=Math.ceil(h+pad*2);\n  var url=\'/patch?x=\'+X+\'&y=\'+Y+\'&w=\'+W+\'&h=\'+H+\'&t=\'+Date.now();\n  return new Promise(function(res,rej){\n    var im=new Image();\n    im.onload=function(){\n      if(!baseReady){base.width=img.naturalWidth;base.height=img.naturalHeight;\n                     bctx.drawImage(img,0,0);baseReady=true;}\n      bctx.drawImage(im,X,Y);          // 只贴这一块，其它像素不动\n      res();\n    };\n    im.onerror=function(){rej(new Error(\'patch fail\'));};\n    im.src=url;\n  }).then(function(){return loadNotes();}).then(function(){draw();});\n}\n/* 一次改动后：优先局部刷新，失败再整页重载 */\nfunction afterEdit(boxes){\n  if(!boxes||!boxes.length) return applyRender();\n  var x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;\n  boxes.forEach(function(b){\n    x0=Math.min(x0,b[0]); y0=Math.min(y0,b[1]);\n    x1=Math.max(x1,b[0]+b[2]); y1=Math.max(y1,b[1]+b[3]);\n  });\n  hold(\'局部刷新…\');\n  return patchRegion(x0,y0,x1-x0,y1-y0).catch(function(){return applyRender();})\n         .finally(release);\n}\nfunction imgXY(e){var r=cv.getBoundingClientRect();return {x:(e.clientX-r.left)/scale, y:(e.clientY-r.top)/scale};}\n\nfunction layout(){\n  if(!img.naturalWidth) return;\n  var cssW=img.naturalWidth*zoomPct/100, cssH=img.naturalHeight*zoomPct/100;\n  cv.width=img.naturalWidth; cv.height=img.naturalHeight;\n  cv.style.width=cssW+\'px\'; cv.style.height=cssH+\'px\';\n  scale=cssW/img.naturalWidth;\n  document.getElementById(\'zl\').textContent=zoomPct+\'%\';\n  draw();\n}\nfunction draw(){\n  if(!img.naturalWidth) return;\n  ctx.clearRect(0,0,cv.width,cv.height);\n  if(showOrig && imgOrig.naturalWidth){ ctx.drawImage(imgOrig,0,0,cv.width,cv.height); return; }\n  ctx.drawImage(baseReady?base:img,0,0,cv.width,cv.height);\n  var lw=Math.max(1,2/scale), i, n;\n  for(i=0;i<NOTES.length;i++){\n    n=NOTES[i]; if(n.u) continue;\n    var on=(selId!==null&&n.id===selId)||selIds.indexOf(n.id)>=0;\n    var aux=(n.sc===\'aux\');\n    ctx.strokeStyle=on?\'rgba(24,201,100,.95)\':(aux?\'rgba(168,96,255,.55)\':\'rgba(77,163,255,.16)\');\n    ctx.lineWidth=on?lw*2:(aux?lw*1.4:lw);\n    ctx.strokeRect(n.x,n.y,n.w,n.h);\n  }\n  for(i=0;i<NOTES.length;i++){\n    n=NOTES[i]; if(!n.u) continue;\n    var on2=(selId!==null&&n.id===selId), g=Math.max(2,n.w*0.35);\n    ctx.strokeStyle=on2?\'rgba(24,201,100,.98)\':\'rgba(255,150,20,.95)\';\n    ctx.lineWidth=on2?lw*2.4:lw*1.8;\n    ctx.strokeRect(n.x-g,n.y-g,n.w+2*g,n.h+2*g);\n  }\n  if(roi){ctx.strokeStyle=\'rgba(255,209,102,.95)\';ctx.lineWidth=lw*2;\n    ctx.setLineDash([8/scale,6/scale]);ctx.strokeRect(roi[0],roi[1],roi[2],roi[3]);ctx.setLineDash([]);}\n  if(drag){ctx.strokeStyle=\'rgba(255,93,93,.95)\';ctx.lineWidth=lw*2;\n    ctx.setLineDash([7/scale,5/scale]);ctx.strokeRect(drag.x,drag.y,drag.w,drag.h);ctx.setLineDash([]);}\n  if(moveDrag){\n    ctx.strokeStyle=\'rgba(255,64,160,.98)\';ctx.lineWidth=lw*2;\n    ctx.setLineDash([6/scale,4/scale]);\n    ctx.strokeRect(moveDrag.b[0]+moveDrag.dx,moveDrag.b[1]+moveDrag.dy,moveDrag.b[2],moveDrag.b[3]);\n    ctx.setLineDash([]);\n  }\n}\nfunction stat(){\n  var nu=0,i; for(i=0;i<NOTES.length;i++) if(NOTES[i].u) nu++;\n  var el=document.getElementById(\'stat\');\n  var t=(ST.source_key===ST.target_key)?(\'1=\'+ST.source_key):(\'1=\'+ST.source_key+\' → 1=\'+ST.target_key);\n  el.innerHTML=\'<b>\'+t+\'</b> \u3000已识别 <b>\'+(NOTES.length-nu)+\'</b> \u3000橙框 <b>\'+nu+\'</b> \u3000转调点 <b>\'+((ST.key_changes||[]).length)+\'</b>\';\n}\nfunction reload(){\n  return new Promise(function(res){\n    img.onload=function(){\n      base.width=img.naturalWidth; base.height=img.naturalHeight;\n      bctx.drawImage(img,0,0); baseReady=true;\n      layout();loadNotes().then(res);\n    };\n    img.onerror=function(){log(\'图片加载失败\');res();};\n    img.src=\'/img?t=\'+Date.now();\n    if(!imgOrig.naturalWidth) imgOrig.src=\'/orig?t=\'+Date.now();\n  });\n}\nfunction loadNotes(){return fetch(\'/state\').then(function(r){return r.json();}).then(function(j){\n  ST=j; NOTES=j.notes||[];\n  var sel=document.getElementById(\'stg\');\n  if(sel&&typeof j.strength===\'number\') sel.value=String(j.strength); syncTgt(); renderKcList(); stat(); draw();\n});}\nfunction bump(){ return applyRender(); }\n\nfunction syncTgt(){\n  var s=document.getElementById(\'tgt\');\n  if(!s.options.length){\n    var h=\'<option value="">同原调（不转）</option>\', i;\n    for(i=0;i<KCKEYS.length;i++) h+=\'<option value="\'+KCKEYS[i]+\'">1=\'+KCKEYS[i]+\'</option>\';\n    s.innerHTML=h;\n  }\n  s.value=(ST.target_key&&ST.target_key!==ST.source_key)?ST.target_key:\'\';\n}\nfunction setTarget(v){\n  var k=v||ST.source_key;\n  api(\'/target\',{key:k}).then(function(r){ if(r.ok){log(\'目标调 → 1=\'+k); return bump();} });\n}\nfunction toggleKc(){\n  kcMode=!kcMode;\n  document.getElementById(\'kcBtn\').className=kcMode?\'on\':\'\';\n  cv.style.cursor=kcMode?\'copy\':\'crosshair\';\n  if(kcMode){log(\'在谱面上点一下放转调点（可以点在一行的半路）\');hideKc();}\n  else{hideKc();}\n}\nfunction showKcPicker(x,y){\n  kcAt={x:x,y:y};\n  document.getElementById(\'kcCard\').style.display=\'block\';\n  document.getElementById(\'kcInfo\').textContent=\'位置 \'+Math.round(x)+\',\'+Math.round(y);\n  var d=document.getElementById(\'kcKeys\'); d.innerHTML=\'\';\n  KCKEYS.forEach(function(k){\n    var b=document.createElement(\'button\'); b.className=\'oc\'; b.textContent=\'1=\'+k;\n    b.onclick=function(){doKeyChange(k);};\n    d.appendChild(b);\n  });\n}\nfunction hideKc(){document.getElementById(\'kcCard\').style.display=\'none\';kcAt=null;}\nfunction doKeyChange(k){\n  if(!kcAt) return;\n  var at=kcAt; hideKc();\n  kcMode=false; document.getElementById(\'kcBtn\').className=\'\'; cv.style.cursor=\'crosshair\';\n  api(\'/keychange\',{x:at.x,y:at.y,key:k}).then(function(r){\n    if(r.ok){log(\'转调点 → 1=\'+k); return bump();}\n  });\n}\nfunction renderKcList(){\n  var ks=ST.key_changes||[], i, h=\'\';\n  document.getElementById(\'kcListCard\').style.display=ks.length?\'block\':\'none\';\n  for(i=0;i<ks.length;i++){\n    h+=\'<div class="plan" style="cursor:default"><span class="ic">♯</span><span class="tx">从 \'+Math.round(ks[i].y)+\' 行高 / x=\'+Math.round(ks[i].x)+\' 起 <b>1=\'+ks[i].key+\'</b></span><button onclick="delKc(\'+i+\')">删除</button></div>\';\n  }\n  document.getElementById(\'kcList\').innerHTML=h;\n}\nfunction delKc(i){\n  api(\'/keychange_del\',{i:i}).then(function(r){ if(r.ok){log(\'已删除转调点\'); return bump();} });\n}\n\nfunction buildDigits(id,cur,cb){\n  var d=document.getElementById(id); d.innerHTML=\'\';\n  \'01234567\'.split(\'\').forEach(function(ch){\n    var b=document.createElement(\'button\'); b.textContent=ch;\n    if(ch===cur) b.className=\'on\';\n    b.onclick=function(){cb(ch);};\n    d.appendChild(b);\n  });\n}\nfunction buildOcts(id,cur,cb){\n  var o=document.getElementById(id); o.innerHTML=\'\';\n  OCTS.forEach(function(p){\n    var b=document.createElement(\'button\'); b.className=\'oc\'+(p[0]===cur?\' on\':\'\');\n    b.textContent=p[1]; b.onclick=function(){cb(p[0]);};\n    o.appendChild(b);\n  });\n}\n\nfunction hitNote(x,y){\n  var i,best=null,bd=1e18;\n  for(i=0;i<NOTES.length;i++){           /* 橙框优先：点在框内直接命中 */\n    var u=NOTES[i]; if(!u.u) continue;\n    var g=Math.max(3,u.w*0.5);\n    if(x>=u.x-g&&x<=u.x+u.w+g&&y>=u.y-g&&y<=u.y+u.h+g) return u;\n  }\n  for(i=0;i<NOTES.length;i++){\n    var n=NOTES[i], cx=n.x+n.w/2, cy=n.y+n.h/2;\n    var d=Math.hypot(cx-x,cy-y);\n    var inside=(x>=n.x-n.w*0.8&&x<=n.x+n.w*1.8&&y>=n.y-n.h*1.2&&y<=n.y+n.h*2.2);\n    if(inside) d*=0.2;\n    if(d<bd){bd=d;best=n;}\n  }\n  if(!best) return null;\n  return (bd<=Math.max(40,best.h*1.4))?best:null;\n}\nfunction showSel(n, keepMulti){\n  hideIns();\n  if(!keepMulti) selIds=[n.id];\n  else if(selIds.indexOf(n.id)<0) selIds.push(n.id);\n  selId=n.id; selData=n; selDigit=n.digit;\n  var card=document.getElementById(\'selCard\');\n  card.style.display=\'block\';\n  if(n.u){\n    card.className=\'card hot\';\n    document.getElementById(\'selTitle\').textContent=\'未识别音（橙框）\';\n    document.getElementById(\'selInfo\').innerHTML=\'#\'+n.id+\' 位置 \'+Math.round(n.x)+\',\'+Math.round(n.y);\n    document.getElementById(\'selDigitTip\').textContent=\'选数字 → 立即回填（再点八度调整）\';\n    buildDigits(\'selDigits\',\'\',function(ch){assignSlot(ch,0);});\n    buildOcts(\'selOcts\',0,function(v){assignSlot(selDigit,v);});\n  }else{\n    card.className=\'card\';\n    document.getElementById(\'selTitle\').textContent=\'选中音符\';\n    document.getElementById(\'selInfo\').innerHTML=\'#\'+n.id+\' 行\'+n.row+\' <b>\'+n.digit+\'</b> 点\'+n.oct+\n      (n.sc===\'aux\'?\' · 辅助\':\' · 主字号\')+(selIds.length>1?(\'\u3000<b>已选 \'+selIds.length+\' 个</b>\'):\'\');\n    document.getElementById(\'selDigitTip\').textContent=\'数字（点一下即改）\';\n    buildDigits(\'selDigits\',n.digit,function(ch){applyEdit(ch,n.oct);});\n    buildOcts(\'selOcts\',n.oct,function(v){applyEdit(n.digit,v);});\n    buildSizes(n);\n  }\n  draw();\n}\nfunction clearSel(){selId=null;selIds=[];document.getElementById(\'selCard\').style.display=\'none\';draw();}\nfunction delSelected(){\n  var ids=selIds.slice();\n  if(!ids.length && selId!==null && selId>=0) ids=[selId];\n  ids=ids.filter(function(x){return x>=0;});\n  if(!ids.length){log(\'没有选中音符\');return;}\n  api(\'/del_many\',{ids:ids}).then(function(r){\n    if(r.ok){log(\'已删除 \'+r.n+\' 个音符\');var bs=ids.map(boxOf).filter(Boolean);\n             selId=null;selIds=[];clearSel();return afterEdit(bs);}\n    log(\'删除失败\');\n  });\n}\nfunction delNote(){\n  if(selId===null||selId<0){log(\'橙框不是音符，不能删\');return;}\n  var id=selId;\n  api(\'/del\',{id:id}).then(function(r){\n    if(r.ok){log(\'已删除 #\'+id);var b=boxOf(id);selId=null;selIds=[];clearSel();return afterEdit(b?[b]:null);}\n    log(\'删除失败\');\n  });\n}\nfunction buildSizes(n){\n  var d=document.getElementById(\'selSizes\'); d.innerHTML=\'\';\n  function mk(label,on,fn){\n    var b=document.createElement(\'button\'); b.textContent=label;\n    if(on) b.className=\'on\';\n    b.onclick=fn; d.appendChild(b);\n  }\n  mk(\'主字号\', n.sc!==\'aux\', function(){setSize([n.id],\'main\');});\n  mk(\'辅助字号\', n.sc===\'aux\', function(){setSize([n.id],\'aux\');});\n  mk(\'本行全主字号\', false, function(){setSizeRow(n.row,\'main\');});\n  mk(\'本行全辅助\', false, function(){setSizeRow(n.row,\'aux\');});\n}\nfunction resel(){\n  var id=selId;\n  hold(\'重新绘制…\');\n  return applyRender().then(function(){\n    var i,n=null;\n    for(i=0;i<NOTES.length;i++) if(NOTES[i].id===id){n=NOTES[i];break;}\n    if(n) showSel(n); else clearSel();\n  }).finally(release);\n}\nfunction setSize(ids,cls){\n  var boxes=ids.map(boxOf).filter(Boolean);\n  api(\'/size\',{ids:ids,cls:cls}).then(function(r){\n    if(r.ok){log(\'字号 → \'+(cls===\'aux\'?\'辅助\':\'主\')+\'（\'+r.n+\' 个）\');\n             return afterEdit(boxes).then(reselSel);}\n  });\n}\nfunction reselSel(){\n  var id=selId,i,n=null;\n  for(i=0;i<NOTES.length;i++) if(NOTES[i].id===id){n=NOTES[i];break;}\n  if(n) showSel(n);\n}\nfunction setSizeRow(row,cls){\n  var boxes=NOTES.filter(function(n){return !n.u && n.row===row;}).map(function(n){return [n.x,n.y,n.w,n.h];});\n  api(\'/size_row\',{row:row,cls:cls}).then(function(r){\n    if(r.ok){log(\'整行字号 → \'+(cls===\'aux\'?\'辅助\':\'主\')+\'（\'+r.n+\' 个）\');\n             return afterEdit(boxes).then(reselSel);}\n  });\n}\nfunction applyEdit(d,o){\n  if(selId===null||selId<0) return;\n  var id=selId, b=boxOf(id);\n  api(\'/edit\',{id:id,digit:d,oct:o}).then(function(r){\n    if(r.ok){log(\'#\'+id+\' → \'+d+\' 点\'+o); return afterEdit(b?[b]:null);}\n  });\n}\nfunction assignSlot(d,o){\n  if(selId===null||selId>=0) return;\n  var slot=-selId-1, b=boxOf(selId), boxes=b?[b]:null;\n  if(b){boxes.push([b[0]-120,b[1]-40,b[2]+240,b[3]+80]);}   // 邻近音被挪动时也覆盖到\n  selDigit=d;\n  api(\'/assign\',{slot:slot,digit:d,oct:o}).then(function(r){\n    if(!r.ok){log(\'回填失败（槽位已变，刷新一下）\');return;}\n    log((r.moved?\'已把旁边歪掉的音挪到这里\':\'橙框回填\')+\' → \'+d+\' 点\'+o+\'（#\'+r.id+\'）\');\n    selId=r.id;\n    return afterEdit(boxes).then(function(){\n      var n=null,i;\n      for(i=0;i<NOTES.length;i++) if(NOTES[i].id===selId){n=NOTES[i];break;}\n      if(n) showSel(n); else clearSel();\n    });\n  });\n}\nfunction jumpNext(){\n  var i,first=null,after=null;\n  for(i=0;i<NOTES.length;i++){\n    if(!NOTES[i].u) continue;\n    if(first===null) first=NOTES[i];\n    if(selId!==null&&NOTES[i].id>selId){after=NOTES[i];break;}\n  }\n  var t=after||first;\n  if(!t){log(\'没有橙框了 🎉\');return;}\n  var cssX=t.x*scale, cssY=t.y*scale;\n  wrap.scrollTop=Math.max(0,cssY-wrap.clientHeight*0.38);\n  wrap.scrollLeft=Math.max(0,cssX-wrap.clientWidth*0.45);\n  showSel(t);\n}\n\nfunction showIns(x,y){\n  insAt={x:x,y:y}; insDigit=\'6\'; insOct=0;\n  document.getElementById(\'insCard\').style.display=\'block\';\n  document.getElementById(\'insInfo\').textContent=\'位置 \'+Math.round(x)+\',\'+Math.round(y);\n  buildDigits(\'insDigits\',\'6\',function(ch){\n    insDigit=ch;\n    var bs=document.getElementById(\'insDigits\').children;\n    for(var i=0;i<bs.length;i++) bs[i].className=(bs[i].textContent===ch?\'on\':\'\');\n    doInsert(insOct);\n  });\n  buildOcts(\'insOcts\',0,function(v){insOct=v; doInsert(v);});\n}\nfunction hideIns(){document.getElementById(\'insCard\').style.display=\'none\';insAt=null;}\nfunction doInsert(oct){\n  if(!insAt) return;\n  api(\'/insert\',{x:insAt.x,y:insAt.y,digit:insDigit,oct:oct}).then(function(r){\n    hideIns();\n    if(r.ok){log(\'插入 \'+insDigit+\' 点\'+oct); return applyRender();}\n  });\n}\n\ncv.addEventListener(\'mousedown\',function(e){\n  var p=imgXY(e);\n  if(e.altKey){\n    var n=hitNote(p.x,p.y);\n    if(n&&!n.u){ moveDrag={id:n.id,x0:p.x,y0:p.y,b:[n.x,n.y,n.w,n.h],dx:0,dy:0}; return; }\n  }\n  drag={px:p.x,py:p.y,x:p.x,y:p.y,w:0,h:0,moved:false,ctrl:(e.ctrlKey||e.metaKey)};\n});\nwindow.addEventListener(\'mousemove\',function(e){\n  if(moveDrag){\n    var pm=imgXY(e);\n    moveDrag.dx=pm.x-moveDrag.x0; moveDrag.dy=pm.y-moveDrag.y0;\n    draw(); return;\n  }\n  if(!drag) return; var p=imgXY(e);\n  drag.x=Math.min(drag.px,p.x);drag.y=Math.min(drag.py,p.y);\n  drag.w=Math.abs(p.x-drag.px);drag.h=Math.abs(p.y-drag.py);\n  // 阈值按**屏幕像素**算：触控板单击时手会微微动，6px 就会被当成拖框，\n  // 结果"单击选不中"。放到 10px，并且要求位移明显才算拖。\n  if(drag.w*scale>10||drag.h*scale>10) drag.moved=true;\n  draw();\n});\nwindow.addEventListener(\'mouseup\',function(e){\n  if(moveDrag){\n    var md=moveDrag; moveDrag=null;\n    if(Math.abs(md.dx)<2&&Math.abs(md.dy)<2){draw();return;}\n    hold(\'移动音符…\');\n    api(\'/move_note\',{id:md.id,dx:Math.round(md.dx),dy:Math.round(md.dy)}).then(function(r){\n      if(r.ok){\n        log(\'已移动 (\'+Math.round(md.dx)+\',\'+Math.round(md.dy)+\')\');\n        var nx=md.b[0]+md.dx, ny=md.b[1]+md.dy;\n        return afterEdit([md.b]).then(function(){\n          // 移动后 id 会重排，按新坐标把这一音重新选中\n          var best=null,bd=1e18,i,nn;\n          for(i=0;i<NOTES.length;i++){nn=NOTES[i]; if(nn.u) continue;\n            var d2=Math.abs(nn.x-nx)+Math.abs(nn.y-ny);\n            if(d2<bd){bd=d2;best=nn;}}\n          if(best&&bd<40) showSel(best);\n        });\n      }\n      log(\'移动失败\');\n    }).finally(release);\n    return;\n  }\n  if(!drag) return; var d=drag; drag=null;\n  // 小抖动也算单击：位移不到 25 屏幕像素就当点击（触控板/手抖）\n  var _small = Math.max(d.w, d.h) * scale < 25;\n  if(!d.moved || _small){\n    if(isDbl){isDbl=false;draw();return;}\n    if(kcMode){showKcPicker(d.px,d.py);draw();return;}\n    var n=hitNote(d.px,d.py);\n    log(\'click (\'+Math.round(d.px)+\',\'+Math.round(d.py)+\') → \'+(n?(\'#\'+n.id):\'空\'));\n    if(n){ showSel(n, d.ctrl); }\n    else if(!d.ctrl){ clearSel(); hideIns(); }   // 单击空白只取消选中\n    draw(); return;\n  }\n  clearTimeout(clickTimer);\n  roi=[d.x,d.y,d.w,d.h]; selId=null; document.getElementById(\'selCard\').style.display=\'none\';\n  api(\'/roi\',{x:d.x,y:d.y,w:d.w,h:d.h}).then(function(r){\n    plans=r.plans; renderPlans(r.plans,r.count); draw();\n  });\n});\ncv.addEventListener(\'dblclick\',function(e){\n  clearTimeout(clickTimer);\n  if(kcMode){return;}\n  var p=imgXY(e);\n  var n=hitNote(p.x,p.y);\n  if(n){showSel(n);return;}\n  showIns(p.x,p.y);\n});\n\nfunction renderPlans(list,count){\n  document.getElementById(\'roiCard\').style.display=\'block\';\n  document.getElementById(\'roiCnt\').textContent=count?(\'框内 \'+count+\' 个已识别\'):\'框内无已识别音符\';\n  var ic=[\'✎\',\'🔍\',\'🧽\',\'➕\',\'✕\'];\n  var h=\'\';\n  for(var i=0;i<list.length;i++){\n    h+=\'<div class="plan" onclick="doPlan(\'+i+\')"><span class="ic">\'+(ic[i]||\'•\')+\'</span><span class="tx">\'+list[i]+\'</span></div>\';\n  }\n  document.getElementById(\'plans\').innerHTML=h;\n}\nfunction doPlan(i){\n  if(!roi) return;\n  var t=plans[i]||\'\';\n  if(t.indexOf(\'已识别音符\')>=0){log(\'框内有音符：请单击选中再改\');return;}\n  if(t.indexOf(\'删除框内所有音符\')>=0){\n    if(!confirm(\'删除框内所有已识别音符？\')) return;\n    api(\'/roi_clear_notes\',{roi:roi}).then(function(r){\n      if(r.ok){log(\'已删除框内 \'+r.n+\' 个音符\');return applyRender();}\n    });\n    return;\n  }\n  if(t.indexOf(\'加强识别\')>=0){\n    api(\'/strengthen\',{mode:\'ocr\',roi:roi}).then(function(r){\n      if(r.ok){\n        log(\'加强识别：读到 \'+(r.found||0)+\' 个字符，新增 \'+(r.added||0)+\' 个音\'\n            +((r.skipped||0)?(\'，已有 \'+r.skipped+\' 个\'):\'\')+\'（原有音符不动）\');\n        return applyRender();\n      }});\n  }else if(t.indexOf(\'加强清除\')>=0){\n    api(\'/strengthen\',{mode:\'erase\',roi:roi}).then(function(r){\n      if(r.ok){log(\'加强清除完成\');reload();}});\n  }else if(t.indexOf(\'强制插入\')>=0){\n    showIns(roi[0]+roi[2]/2, roi[1]+roi[3]/2);\n  }else{clearRoi();}\n}\nfunction clearRoi(){roi=null;plans=[];document.getElementById(\'roiCard\').style.display=\'none\';draw();}\nfunction undo(){\n  api(\'/undo\',{}).then(function(r){\n    if(r.ok){log(\'已回退\');return applyRender();} log(\'没有可回退的步骤\');\n  });\n}\nfunction setStrength(level){\n  var lb={0:\'严格\',1:\'标准\',2:\'加强\'}[level]||level;\n  busy(true,\'按「\'+lb+\'」重新识别…\');\n  document.querySelectorAll(\'button\').forEach(function(x){x.disabled=true;});\n  fetch(\'/strength\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},\n                     body:JSON.stringify({level:parseInt(level,10)})})\n    .then(function(r){return r.json();}).then(function(r){\n      if(r.ok){log(\'识别强度 → \'+lb+\'：音符 \'+r.pairs+\'，橙框 \'+r.uncovered);return applyRender();}\n      log(\'切换强度失败\');\n    }).finally(function(){\n      busy(false);\n      document.querySelectorAll(\'button\').forEach(function(x){x.disabled=false;});\n    });\n}\nfunction saveLib(){\n  api(\'/save\',{}).then(function(r){\n    if(r.ok){log(\'已保存到库缓存（下次打开秒开）\');toast(\'已保存\');}\n    else log(\'保存失败：这一页不是从乐谱库打开的\');\n  });\n}\nfunction toast(s){\n  var t=document.getElementById(\'toastx\');\n  if(!t){\n    t=document.createElement(\'div\'); t.id=\'toastx\';\n    t.style.cssText=\'position:fixed;right:20px;top:70px;background:#1b222b;color:#e6e9ee;\'+\n      \'border:1px solid #2b333d;border-radius:9px;padding:9px 14px;font-size:13px;z-index:9999\';\n    document.body.appendChild(t);\n  }\n  t.textContent=s; t.style.display=\'block\';\n  clearTimeout(t._h); t._h=setTimeout(function(){t.style.display=\'none\';},2200);\n}\nfunction zoom(d){zoomPct=Math.max(20,Math.min(400,zoomPct+(d>0?20:-20)));layout();}\nfunction fitWidth(){if(!img.naturalWidth)return;zoomPct=Math.max(10,Math.round(wrap.clientWidth*0.97/img.naturalWidth*100));layout();}\nwindow.addEventListener(\'resize\',function(){if(img.naturalWidth)layout();});\nwindow.addEventListener(\'keydown\',function(e){\n  if(e.key===\'Control\'&&!showOrig){showOrig=true;draw();}\n  if((e.key===\'Delete\'||e.key===\'Backspace\')&&!insAt&&!kcMode){\n    if(selIds.some(function(x){return x>=0;})){e.preventDefault();delSelected();}\n    else if(selId!==null&&selId>=0){e.preventDefault();delNote();}\n  }\n  if(e.key===\'Escape\'){clearSel();hideIns();}\n  if(insAt && /^[0-7]$/.test(e.key)){\n    e.preventDefault(); insDigit=e.key; doInsert(insOct); return;\n  }\n  if(insAt && (e.key===\'ArrowUp\'||e.key===\'ArrowDown\')){\n    e.preventDefault();\n    insOct=Math.max(-2,Math.min(2,insOct+(e.key===\'ArrowUp\'?1:-1)));\n    document.getElementById(\'insInfo\').textContent=\'八度 \'+insOct;\n  }\n  if((e.ctrlKey||e.metaKey)&&(e.key===\'z\'||e.key===\'Z\')){e.preventDefault();undo();}\n});\nwindow.addEventListener(\'keyup\',function(e){ if(e.key===\'Control\'&&showOrig){showOrig=false;draw();} });\nwindow.addEventListener(\'blur\',function(){ if(showOrig){showOrig=false;draw();} });\nlog(\'页面已加载 build __BUILD__\');\nreload();\n</script></body></html>\n'


def _read_page(path: str, fallback: str) -> str:
    try:
        return open(path, encoding="utf8").read()
    except Exception:
        return fallback


def _build_stamp() -> str:
    """页面构建时间 = work/*.html 与 jp/*.py 里最新的修改时间。

    放在页面上就是为了让"我到底看的是不是最新版"一眼能确认。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    latest = 0.0
    for sub, exts in (("work", (".html",)), ("jp", (".py",))):
        d = os.path.join(root, sub)
        try:
            for f in os.listdir(d):
                if f.endswith(exts):
                    latest = max(latest, os.path.getmtime(os.path.join(d, f)))
        except Exception:
            pass
    if latest <= 0:
        return time.strftime("%Y-%m-%d %H:%M")
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(latest))


BUILD = _build_stamp()

LIBPAGE = _read_page(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "work", "lib_page.html"),
                     "<h1>缺少 work\\lib_page.html</h1>")
NOPAGE = ("<!doctype html><meta charset='utf-8'><body style='font:16px/1.8 system-ui;"
          "background:#0f1216;color:#e6e9ee;padding:40px'>"
          "<h1>还没有打开乐谱</h1><p><a style='color:#4da3ff' href='/lib'>→ 去乐谱库导入/选择</a></p>")


class Job:
    """后台任务：打开乐谱 / 批量处理都可能跑几十秒，不能卡住请求。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.running = False
            self.kind = ""
            self.msg = ""
            self.done = 0
            self.total = 0
            self.error = ""

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snap(self) -> dict:
        with self.lock:
            return dict(running=self.running, kind=self.kind, msg=self.msg,
                        done=self.done, total=self.total, error=self.error)


JOB = Job()
ACTIVE = {"st": None, "lib_id": None}
ALOCK = threading.Lock()


def start_job(kind: str, target) -> bool:
    """target(job) 在后台线程里跑。已有任务在跑就返回 False。"""
    with JOB.lock:
        if JOB.running:
            return False
        JOB.running = True
        JOB.kind = kind
        JOB.msg = "开始"
        JOB.done = 0
        JOB.total = 1
        JOB.error = ""

    def run():
        try:
            target(JOB)
            JOB.set(msg="完成")
        except Exception as e:
            JOB.set(error="%s: %s" % (type(e).__name__, e))
        finally:
            JOB.set(running=False)

    threading.Thread(target=run, daemon=True).start()
    return True


def open_score(job, lib_id: str):
    import jp.library as LIB
    it = LIB.get(lib_id)
    if not it:
        raise RuntimeError("乐谱不存在")
    job.set(msg="读取 OCR 文本", done=0, total=1)
    txt_path = it.get("ocr_txt") or LIB.ocr_path_for_id(it["id"])
    if not __import__("os").path.exists(txt_path):
        job.set(msg="生成 OCR 文本（较慢）")
        LIB.ensure_ocr(it)
    job.set(msg="构建页面对齐（约 10–30 秒）")
    key = it.get("key") or LIB.detect_key(open(txt_path, encoding="utf8").read()) or "C"
    _st = it.get("strength")
    if _st is None:
        _st = 0 if it.get("manual") else 1
    st = State(it["image"], txt_path, key, title=it.get("title") or it.get("song") or "",
               cache_key=lib_id, strength=int(_st))
    if it.get("target"):
        pc = M.parse_key(it["target"])
        if pc is not None:
            st.target_pc = int(pc)
    # 库里检测到「中途转调」→ 直接在纠错页种好锚点，用户可拖/删/改
    if it.get("multi") and (it.get("keys") or []):
        try:
            txt = open(txt_path, encoding="utf8").read()
            nlines = max(1, len(txt.splitlines()))
            H = float(st.page.gray.shape[0])
            for k in it["keys"][1:]:
                pc = M.parse_key(k.get("key", ""))
                if pc is None:
                    continue
                st.key_changes.append(dict(x=0.0, y=H * float(k.get("line", 0)) / nlines,
                                           key_pc=int(pc)))
            if st.key_changes:
                st.refresh()
        except Exception:
            pass
    with ALOCK:
        ACTIVE["st"] = st
        ACTIVE["lib_id"] = lib_id
    job.set(done=1, msg="已打开")


def _ai_fill_one(LIB, lib_id: str) -> dict:
    """只把这一页**已有的 OCR 文字**发给 LLM 补标题/调号；图片一个字节都不发。"""
    import jp.llm as LLM
    if not LLM.available():
        raise RuntimeError("未配置 AI：请在 设置 → AI 配置 里填写 base/model/key")
    it = LIB.get(lib_id)
    if not it:
        raise RuntimeError("乐谱不存在")
    tp = it.get("ocr_txt") or LIB.ocr_path_for_id(it["id"])
    if not os.path.exists(tp):
        raise RuntimeError("这一页还没有 OCR 文本，先跑 OCR")
    txt = open(tp, encoding="utf8").read()
    cfg = LLM.config()
    meta = LLM.identify(txt, use_search=bool(cfg.get("use_search", True)))
    upd = {}
    if meta.get("title") and not it.get("title"):
        upd["title"] = meta["title"]
    k = (meta.get("key") or "").strip()
    if k and M.parse_key(k) is not None and not it.get("key"):
        upd["key"] = KEY_NAMES[int(M.parse_key(k)) % 12]
    if upd:
        LIB.update(lib_id, **upd)
    return meta


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
                ctype = "application/json; charset=utf-8"
            try:
                self.send_response(code)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                return
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return {}

        def _multipart(self):
            """极简 multipart 解析：只要能收 <input type=file> 传上来的图片。"""
            import re as _re
            ctype = self.headers.get("Content-Type") or ""
            n = int(self.headers.get("Content-Length") or 0)
            if "boundary=" not in ctype:
                return [], {}
            bnd = ctype.split("boundary=", 1)[1].strip().strip('"').encode()
            raw = self.rfile.read(n)
            files, fields = [], {}
            for part in raw.split(b"--" + bnd):
                if b"\r\n\r\n" not in part:
                    continue
                head, body = part.split(b"\r\n\r\n", 1)
                if body.endswith(b"\r\n"):
                    body = body[:-2]
                hs = head.decode("utf8", "replace")
                if "filename=" not in hs:
                    m = _re.search(r'name="([^"]+)"', hs)
                    if m:
                        fields[m.group(1)] = body.decode("utf8", "replace")
                    continue
                m = _re.search(r'filename="([^"]*)"', hs)
                fn = (m.group(1) if m else "").strip()
                if fn:
                    files.append((fn, body))
            return files, fields

        def _st(self):
            with ALOCK:
                return ACTIVE["st"]

        def do_GET(self):
            import jp.library as LIB
            st = self._st()
            if self.path.startswith("/api/lib"):
                _all = LIB.load()
                _trash_n = sum(1 for x in _all.values() if x.get("deleted"))
                items = sorted([x for x in _all.values() if not x.get("deleted")],
                               key=lambda x: ((x.get("song") or ""),
                                              x.get("order") if isinstance(x.get("order"), (int, float)) else 1e12,
                                              (x.get("image") or "")))
                # 老记录没有 keys 字段：有 OCR 文本就顺手补上（只读几个小文本，很快）
                for it in items:
                    if it.get("keys"):
                        continue
                    tp = it.get("ocr_txt") or LIB.ocr_path_for_id(it["id"])
                    if not os.path.exists(tp):
                        continue
                    try:
                        ks = LIB.detect_keys(open(tp, encoding="utf8").read())
                    except Exception:
                        ks = []
                    if ks:
                        it["keys"] = ks
                        it["multi"] = len(ks) > 1
                        if not it.get("key"):
                            it["key"] = ks[0]["key"]
                        LIB.update(it["id"], keys=ks, multi=len(ks) > 1,
                                   **({"key": ks[0]["key"]} if not it.get("key") else {}))
                return self._send(200, dict(stats=LIB.stats(), items=items,
                                            groups=LIB.groups(), trash_n=_trash_n,
                                            active=ACTIVE["lib_id"],
                                            llm=__import__("jp.llm", fromlist=["x"]).available(),
                                            ocr=XO.available(),
                                            root=LIB.ROOT))
            if self.path.startswith("/api/preview"):
                qs = __import__("urllib.parse", fromlist=["x"]).parse_qs(
                    self.path.split("?", 1)[1] if "?" in self.path else "")
                pid = (qs.get("id") or [""])[0]
                pp = LIB.ensure_preview(pid) if pid else None
                if not pp or not os.path.exists(pp):
                    return self._send(404, b"", "image/jpeg")
                return self._send(200, open(pp, "rb").read(), "image/jpeg")
            if self.path.startswith("/api/llm"):
                import jp.llm as LLM
                return self._send(200, LLM.public_config())
            if self.path.startswith("/api/trash"):
                items = sorted(LIB.trash(), key=lambda x: x.get("deleted_at") or "")
                return self._send(200, dict(items=items))
            if self.path.startswith("/api/thumb"):
                qs = __import__("urllib.parse", fromlist=["x"]).parse_qs(
                    self.path.split("?", 1)[1] if "?" in self.path else "")
                tid = (qs.get("id") or [""])[0]
                tp = LIB.ensure_thumb(tid) if tid else None
                if not tp or not os.path.exists(tp):
                    return self._send(404, b"", "image/jpeg")
                return self._send(200, open(tp, "rb").read(), "image/jpeg")
            if self.path.startswith("/api/sources"):
                return self._send(200, dict(sources=LIB.sources(), root=LIB.ROOT))
            if self.path.startswith("/api/version"):
                return self._send(200, dict(build=BUILD, page_len=len(PAGE),
                                            lib_len=len(LIBPAGE),
                                            song=(st.song if st else ""),
                                            notes=(len(st.pairs) if st else 0)))
            if self.path.startswith("/api/job"):
                return self._send(200, JOB.snap())
            if self.path.startswith("/api/log"):
                tail = []
                lp = os.path.join("out", "roi_server.log")
                if os.path.exists(lp):
                    tail = open(lp, encoding="utf8", errors="replace").read().split("\n")[-40:]
                return self._send(200, dict(lines=[x for x in tail if x]))
            if self.path.startswith("/lib"):
                return self._send(200, LIBPAGE.replace("__BUILD__", BUILD).encode(),
                                  "text/html; charset=utf-8")
            if st is None:
                return self._send(200, NOPAGE.encode(), "text/html; charset=utf-8")
            if self.path.startswith("/orig"):
                with st.lock:
                    return self._send(200, st.orig_png, "image/png")
            if self.path.startswith("/patch"):
                import urllib.parse as _up
                qs = _up.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
                try:
                    x, y, w, h = [int(float((qs.get(k) or ["0"])[0])) for k in ("x", "y", "w", "h")]
                except Exception:
                    return self._send(400, b"", "image/png")
                data = st.patch_png(x, y, w, h)
                if data is None:
                    return self._send(404, b"", "image/png")
                return self._send(200, data, "image/png")
            if self.path.startswith("/img"):
                with st.lock:
                    if not st.png:
                        st.png = st._render()
                    return self._send(200, st.png, "image/png")
            if self.path.startswith("/notes"):
                return self._send(200, st.notes())
            if self.path.startswith("/state"):
                with st.lock:
                    return self._send(200, st.state())
            return self._send(200, PAGE.replace("__SONG__", st.song)
                              .replace("__BUILD__", BUILD).encode(),
                              "text/html; charset=utf-8")

        def do_POST(self):
            import jp.library as LIB
            pth = self.path
            # ---- 上传导入（浏览器文件/文件夹选择器）必须先解 multipart，不能先读 JSON
            if pth.startswith("/api/import_manual") or pth.startswith("/api/import_auto"):
                auto = pth.startswith("/api/import_auto")
                files, fields = self._multipart()
                if not files:
                    return self._send(200, dict(ok=False, error="没有收到文件"))
                title = (fields.get("title") or "").strip()
                key = (fields.get("key") or "").strip()
                target = (fields.get("target") or "").strip()
                gid = (fields.get("group_id") or "").strip()

                def run_import(job):
                    tag = "AI 批量导入" if auto else "人工导入"
                    job.set(total=1, done=0, msg="%s：收文件…" % tag)
                    res = LIB.ingest_files(files, title=title, key=key, target=target,
                                           group_id=gid,
                                           log=lambda m: job.set(msg="%s：%s" % (tag, str(m)[:90])))
                    ids = res.get("ids") or []
                    if not ids:
                        job.set(done=1, msg="%s：没有新增页" % tag)
                        return
                    job.set(total=max(1, len(ids)), done=0,
                            msg="%s：OCR %d 页…" % (tag, len(ids)))
                    LIB.ocr_batch_ids(ids, mode="max",
                                      log=lambda m: job.set(msg=str(m)[:110]))
                    for i in ids:
                        upd = {}
                        it = LIB.get(i) or {}
                        if title:
                            upd["title"] = title
                        if key:
                            upd["key"] = key
                        if target:
                            upd["target"] = target
                        upd["no_ai"] = True
                        upd["manual"] = not auto
                        if it.get("ocr_txt"):
                            upd["status"] = "done" if it.get("out") else "ocr"
                        if upd:
                            LIB.update(i, **upd)
                    if auto:
                        job.set(msg="%s：AI 认歌名（%d 页）…" % (tag, len(ids)))
                        done = 0
                        for i in ids:
                            try:
                                _ai_fill_one(LIB, i)
                            except Exception as e:
                                job.set(msg="AI 失败（%s），继续" % str(e)[:60])
                            done += 1
                            job.set(done=done)
                        job.set(msg="%s：自动分份…" % tag)
                        LIB.auto_group_by_title(log=lambda m: job.set(msg=str(m)[:110]))
                    job.set(done=max(1, len(ids)), msg="%s：完成 %d 页" % (tag, len(ids)))

                return self._send(200, dict(ok=start_job("import", run_import)))
            if pth.startswith("/api/upload"):
                files, fields = self._multipart()
                if not files:
                    return self._send(200, dict(ok=False, error="没有收到文件"))
                song = (fields.get("song") or "").strip()
                key = (fields.get("key") or "").strip()
                sid = (fields.get("source_id") or "").strip()
                if files and not sid:
                    src = LIB.add_source(fields.get("folder") or ("上传 %d 个文件" % len(files)),
                                         "upload", note=song)
                    sid = src["id"]
                added, dup, bad = 0, 0, 0
                ids, badnames = [], []
                for fn, data in files:
                    base = os.path.basename(fn)
                    if not base.lower().endswith(LIB.IMG_EXT):
                        bad += 1
                        badnames.append(base + "（格式不支持）")
                        continue
                    if "(原调" in base:
                        bad += 1
                        badnames.append(base + "（对照图已跳过）")
                        continue
                    try:
                        # 用相对路径里的目录名当歌名（webkitdirectory 会带上）
                        parts = fn.replace("\\", "/").split("/")
                        gname = song or (parts[-2] if len(parts) >= 2 else "")
                        i, created = LIB.import_uploaded(parts[-1], data, song=gname,
                                                         source_id=sid, key=key)
                        ids.append(i)
                        added += 1 if created else 0
                        dup += 0 if created else 1
                    except Exception as e:
                        bad += 1
                        badnames.append("%s（%s）" % (base, str(e)[:40]))
                return self._send(200, dict(ok=True, added=added, dup=dup, bad=bad,
                                            badnames=badnames[:20], source=sid, ids=ids))
            d = self._body()
            # ---- 乐谱库（不需要已打开的谱面）
            if pth.startswith("/api/import"):
                root = str(d.get("root") or "").strip('" ')
                if not root or not (os.path.isdir(root) or os.path.isfile(root)):
                    return self._send(200, dict(ok=False, error="路径不存在: %s" % root))
                added, seen, total = LIB.scan(root, str(d.get("key") or ""))
                # 登记成导入源，之后「更新」按钮能重新扫这个目录
                if not any((x.get("note") or "") == root for x in LIB.sources()):
                    LIB.add_source(os.path.basename(root.rstrip("\\/")) or root,
                                   "path", note=root)
                return self._send(200, dict(ok=True, added=added, seen=seen, total=total))
            if pth.startswith("/api/group_new"):
                g = LIB.group_new(str(d.get("name") or ""))
                return self._send(200, dict(ok=True, group=g))
            if pth.startswith("/api/llm_save"):
                import jp.llm as LLM
                try:
                    cfg = LLM.save_config(str(d.get("base") or ""),
                                          str(d.get("model") or ""),
                                          str(d.get("key") or ""),
                                          use_search=(None if d.get("use_search") is None
                                                      else bool(d.get("use_search"))),
                                          workers=(None if not d.get("workers")
                                                   else int(d.get("workers"))))
                    return self._send(200, dict(ok=True, config=cfg))
                except Exception as e:
                    return self._send(200, dict(ok=False, error=str(e)[:200]))
            if pth.startswith("/api/llm_test"):
                import jp.llm as LLM
                return self._send(200, LLM.test_connection())
            if pth.startswith("/api/delete"):
                n = LIB.soft_delete([str(x) for x in (d.get("ids") or [])])
                return self._send(200, dict(ok=True, n=n))
            if pth.startswith("/api/restore"):
                n = LIB.restore([str(x) for x in (d.get("ids") or [])])
                return self._send(200, dict(ok=True, n=n))
            if pth.startswith("/api/purge"):
                n = LIB.purge([str(x) for x in (d.get("ids") or [])])
                return self._send(200, dict(ok=True, n=n))
            if pth.startswith("/api/dedupe_exact"):
                def run_exact(job):
                    job.set(total=1, done=0, msg="按图片字节/OCR 文本找完全重复")
                    res = LIB.dedupe_exact(log=lambda x: job.set(msg=str(x)[:120]))
                    job.set(done=1, msg="去掉 %d 页重复（%d 组，每组留 1 页）"
                            % (res["removed"], res["groups"]))

                return self._send(200, dict(ok=start_job("dedupe", run_exact)))
            if pth.startswith("/api/dupes"):
                def run_dupes(job):
                    job.set(total=1, done=0, msg="比对图片指纹")
                    res = LIB.mark_dupes(log=lambda s: job.set(msg=s))
                    job.set(done=1, msg="疑似重复 %d 页" % res["dupes"])

                return self._send(200, dict(ok=start_job("dupes", run_dupes)))
            if pth.startswith("/api/group_by_title"):
                def run_bt(job):
                    job.set(total=1, done=0, msg="按 AI 标题分份")
                    res = LIB.auto_group_by_title(log=lambda x: job.set(msg=x))
                    job.set(done=1, msg="分成 %d 份，共 %d 页" % (res["groups"], res["pages"]))

                return self._send(200, dict(ok=start_job("group_title", run_bt)))
            if pth.startswith("/api/titles_clear"):
                n = LIB.clear_titles(ids=[str(x) for x in (d.get("ids") or [])] or None,
                                     group=str(d.get("group") or ""),
                                     all_pages=bool(d.get("all")))
                return self._send(200, dict(ok=True, n=n))
            if pth.startswith("/api/title"):
                it = LIB.update(str(d["id"]), title=str(d.get("title") or ""))
                return self._send(200, dict(ok=it is not None, item=it))
            if pth.startswith("/api/edits"):
                import jp.edits as ED
                if "enabled" in d:
                    ED.set_enabled(bool(d.get("enabled")))
                return self._send(200, ED.stats())
            if pth.startswith("/api/ocr_all"):
                mode = str(d.get("ocr_mode") or "serial")
                ids = [str(x) for x in (d.get("ids") or [])] or list(LIB.load())

                def run_ocr_all(job):
                    todo = [i for i in ids
                            if not ((LIB.get(i) or {}).get("ocr_txt")
                                    and os.path.exists((LIB.get(i) or {}).get("ocr_txt") or ""))]
                    job.set(total=max(1, len(todo)), done=0,
                            msg="批量 OCR：%d 页，方式 %s" % (len(todo), mode))
                    res = LIB.ocr_batch_ids(todo, mode=mode,
                                            log=lambda s: job.set(msg=str(s)[:120]))
                    job.set(done=res.get("total", 0),
                            msg="OCR 完成 %d 页" % res.get("done", 0))

                return self._send(200, dict(ok=start_job("ocr", run_ocr_all)))
            if pth.startswith("/api/reorder_pages"):
                n = LIB.reorder_pages(str(d.get("group") or ""),
                                      [str(x) for x in (d.get("ids") or [])])
                return self._send(200, dict(ok=True, n=n))
            if pth.startswith("/api/group_order"):
                n = LIB.group_reorder([str(x) for x in (d.get("ids") or [])])
                return self._send(200, dict(ok=True, n=n))
            if pth.startswith("/api/group_titles"):
                res = LIB.aggregate_group_titles()
                return self._send(200, dict(ok=True, **res))
            if pth.startswith("/api/group_toggle"):
                g = LIB.group_set(str(d.get("id") or ""), collapsed=bool(d.get("collapsed")))
                return self._send(200, dict(ok=g is not None, group=g))
            if pth.startswith("/api/group_rename"):
                g = LIB.group_set(str(d.get("id") or ""), name=str(d.get("name") or ""))
                return self._send(200, dict(ok=g is not None, group=g))
            if pth.startswith("/api/group_del"):
                n = LIB.group_del(str(d.get("id") or ""))
                return self._send(200, dict(ok=True, moved=n))
            if pth.startswith("/api/move"):
                n = LIB.move_to([str(x) for x in (d.get("ids") or [])],
                                str(d.get("group") or ""))
                return self._send(200, dict(ok=True, n=n))
            if pth.startswith("/api/rescan"):
                # 把「按路径登记」的那些目录重新扫一遍，捡新文件（只登记，不烧 GPU）
                LIB.ensure_sources_from_origins()
                added = seen = 0
                for src in LIB.sources():
                    if src.get("kind") != "path":
                        continue
                    root = (src.get("note") or "").strip()
                    if root and (os.path.isdir(root) or os.path.isfile(root)):
                        a, s2, _t = LIB.scan(root)
                        added += a
                        seen += s2
                return self._send(200, dict(ok=True, added=added, seen=seen,
                                            total=len(LIB.load())))
            if pth.startswith("/api/group_auto"):
                def run_group(job):
                    job.set(total=1, done=0, msg="按时间+歌词分组")
                    res = LIB.auto_group(log=lambda s: job.set(msg=s))
                    job.set(done=1, msg="分成 %d 份，共 %d 页" % (res["groups"], res["pages"]))

                return self._send(200, dict(ok=start_job("group", run_group)))
            if pth.startswith("/api/source_add"):
                ent = LIB.add_source(str(d.get("name") or "").strip() or "(未命名)",
                                     str(d.get("kind") or "path"),
                                     note=str(d.get("note") or ""))
                return self._send(200, dict(ok=True, source=ent))
            if pth.startswith("/api/source_del"):
                return self._send(200, dict(ok=LIB.del_source(str(d.get("id") or ""))))
            if pth.startswith("/api/archive"):
                def run_arch(job):
                    ids = list(LIB.load())
                    job.set(total=len(ids), done=0, msg="把图片复制进本地库")
                    for k, i in enumerate(ids, 1):
                        try:
                            LIB.archive_existing(i)
                        except Exception:
                            pass
                        job.set(done=k, msg="已归档 %d/%d" % (k, len(ids)))
                    job.set(msg="完成：%d 页已经复制进 library\\ 目录" % len(ids))

                return self._send(200, dict(ok=start_job("archive", run_arch)))
            if pth.startswith("/api/save"):
                st = self._st()
                ok = bool(st and st.save_cache())
                return self._send(200, dict(ok=ok))
                def run_dedupe(job):
                    ids = list(LIB.load())
                    job.set(total=len(ids), done=0, msg="检查有没有已转调过的图")
                    n = 0
                    for k, i in enumerate(ids, 1):
                        it = LIB.get(i) or {}
                        m = LIB.red_metrics(it.get("image", ""))
                        dup = bool(m and m["ratio"] > 0.06 and m["n"] > 3000)
                        if dup != bool(it.get("dup")):
                            LIB.update(i, dup=dup)
                        n += 1 if dup else 0
                        job.set(done=k, msg="发现已转调 %d 张（第 %d/%d）" % (n, k, len(ids)))
                    job.set(msg="完成：%d 张已转调，批量时会跳过" % n)

                return self._send(200, dict(ok=start_job("dedupe", run_dedupe)))
            if pth.startswith("/api/reorder"):
                ids = [str(x) for x in (d.get("ids") or [])]
                for k, i in enumerate(ids):
                    LIB.update(i, order=k)
                return self._send(200, dict(ok=True, n=len(ids)))
            if pth.startswith("/api/classify"):
                def run_classify(job):
                    import jp.llm as LLM
                    if not LLM.available():
                        raise RuntimeError("未配置 AI：请在 设置 → AI 配置 里填写 base/model/key")
                    only = {str(x) for x in (d.get("ids") or [])}
                    cand = []
                    for i, it in LIB.load().items():
                        if only and i not in only:
                            continue
                        if it.get("title") and not only:
                            continue
                        tp = it.get("ocr_txt") or LIB.ocr_path_for_id(it["id"])
                        if os.path.exists(tp):
                            cand.append(i)
                    if not cand:
                        raise RuntimeError("没有需要分类的页（都有标题，或都还没 OCR）")
                    cfg = LLM.config()
                    workers = max(1, min(16, int(cfg.get("workers", 10) or 10)))
                    job.set(total=len(cand), done=0,
                            msg="AI 分类：%d 页，%d 并发%s" %
                                (len(cand), workers, "，联网搜歌名" if cfg.get("use_search", True) else ""))
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    def _one(i):
                        try:
                            _ai_fill_one(LIB, i)
                            return i, None
                        except Exception as e:
                            return i, str(e)[:120]

                    done = 0
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futs = [ex.submit(_one, i) for i in cand]
                        for k, f in enumerate(as_completed(futs), 1):
                            _i, err = f.result()
                            if not err:
                                done += 1
                            job.set(done=k, msg="已分类 %d/%d%s" % (done, k, ("（失败：%s）" % err) if err else ""))

                return self._send(200, dict(ok=start_job("classify", run_classify)))
            if pth.startswith("/api/meta"):
                it = LIB.update(str(d["id"]), **{k: d[k] for k in
                                                ("title", "key", "target", "multi", "dup", "order")
                                                if k in d})
                return self._send(200, dict(ok=it is not None, item=it))
            if pth.startswith("/api/open"):
                lib_id = str(d["id"])
                return self._send(200, dict(ok=start_job("open", lambda j: open_score(j, lib_id))))
            if pth.startswith("/api/process_all"):
                ids = [str(x) for x in (d.get("ids") or [])] or list(LIB.load())
                mode = str(d.get("mode") or "render")
                if mode in ("ocr", "render") and not d.get("force"):
                    keep = [i for i in ids if not (LIB.get(i) or {}).get("dup")]
                    if len(keep) != len(ids):
                        skip = len(ids) - len(keep)
                        ids = keep
                    else:
                        skip = 0
                else:
                    skip = 0

                def run_all(job):
                    import jp.llm as LLM
                    if mode == "ai":
                        if not LLM.available():
                            raise RuntimeError("未配置 AI：请在 设置 → AI 配置 里填写 base/model/key")
                        # 只挑还没补齐、并且已经有 OCR 文本的页，避免白发 token
                        keep = []
                        for i in ids:
                            it = LIB.get(i) or {}
                            tp = it.get("ocr_txt") or (
                                LIB.ocr_path_for_id(it["id"]) if it else "")
                            if it.get("title") and it.get("key"):
                                continue
                            if tp and os.path.exists(tp):
                                keep.append(i)
                        ids[:] = keep
                        if not ids:
                            raise RuntimeError("没有可补信息的页（要么已齐，要么还没 OCR）")
                    job.set(total=len(ids), done=0, msg="")
                    fails = 0
                    for k, i in enumerate(ids, 1):
                        it = LIB.get(i) or {}
                        name = it.get("title") or it.get("song") or i
                        job.set(msg="(%d/%d) %s  %s" % (k, len(ids), name, mode))
                        try:
                            if mode == "ocr":            # 只提取文字，本地跑，不出图
                                LIB.do_ocr(i)
                            elif mode == "ai":           # 只把已有文字发给 LLM
                                _ai_fill_one(LIB, i)
                            else:                        # 转调出图（必要时先 OCR）
                                LIB.process(i, log=lambda s: job.set(msg=s))
                        except Exception as e:
                            fails += 1
                            if mode in ("ocr", "render"):   # AI 失败不该把谱面标成出错
                                LIB.update(i, status="error")
                            job.set(msg="失败: %s" % e)
                        job.set(done=k)
                    if fails:
                        job.set(msg="完成：%d 个失败" % fails)
                    elif skip:
                        job.set(msg="完成（跳过 %d 张已转调的图）" % skip)

                return self._send(200, dict(ok=start_job(mode, run_all)))
            if pth.startswith("/api/process"):
                lib_id = str(d["id"])
                mode = str(d.get("mode") or "render")

                def run_one(job):
                    job.set(total=1, done=0, msg=mode)
                    if mode == "ocr":
                        LIB.do_ocr(lib_id)
                    elif mode == "ai":
                        _ai_fill_one(LIB, lib_id)
                    else:
                        LIB.process(lib_id, log=lambda s: job.set(msg=s))
                    job.set(done=1)

                return self._send(200, dict(ok=start_job(mode, run_one)))
            if pth.startswith("/api/ai_meta"):
                lib_id = str(d["id"])
                it = LIB.get(lib_id)
                if not it:
                    return self._send(200, dict(ok=False, error="乐谱不存在"))
                try:
                    meta = _ai_fill_one(LIB, lib_id)
                except Exception as e:
                    return self._send(200, dict(ok=False, error=str(e)[:200]))
                return self._send(200, dict(ok=True, meta=meta, item=LIB.get(lib_id)))
            # ---- 已打开的谱面上的编辑操作
            st = self._st()
            if st is None:
                return self._send(409, dict(ok=False, error="没有打开的乐谱"))
            with st.lock:
                if pth.startswith("/select"):
                    i = st.nearest(float(d["x"]), float(d["y"]))
                    if i is None:
                        return self._send(200, {"ok": False})
                    s, o = st.pairs[i]
                    return self._send(200, dict(ok=True, id=i, row=int(s.row), digit=str(o.digit),
                                                oct=int(o.octave),
                                                box=[float(v) for v in s.box]))
                if pth.startswith("/roi_clear_notes"):
                    return self._send(200, {"ok": True,
                                            "n": st.clear_notes_in_roi(tuple(d["roi"]))})
                if pth.startswith("/roi"):
                    roi = (d["x"], d["y"], d["w"], d["h"])
                    return self._send(200, dict(plans=st.plan_for(roi), count=len(st.in_roi(roi))))
                if pth.startswith("/strengthen"):
                    info = st.strengthen(d.get("mode", "ocr"), d["roi"])
                    return self._send(200, dict(ok=True, **info))
                if pth.startswith("/edit"):
                    st.edit_note(int(d["id"]), str(d.get("digit", "0")), int(d.get("oct", 0)))
                    return self._send(200, {"ok": True})
                if pth.startswith("/strength"):
                    ok = st.set_strength(int(d.get("level", 1)))
                    return self._send(200, {"ok": ok, "level": st.strength,
                                            "pairs": len(st.pairs),
                                            "uncovered": len(st.uncovered)})
                if pth.startswith("/move_note"):
                    ok = st.move_note(int(d.get("id", -1)), float(d.get("dx", 0)),
                                      float(d.get("dy", 0)))
                    return self._send(200, {"ok": ok, "bbox": st.last_bbox})
                if pth.startswith("/del_many"):
                    return self._send(200, {"ok": True,
                                            "n": st.del_notes(d.get("ids") or [])})
                if pth.startswith("/del"):
                    return self._send(200, {"ok": st.del_note(int(d.get("id", -1)))})
                if pth.startswith("/insert"):
                    st.insert_note(float(d["x"]), float(d["y"]),
                                   str(d.get("digit", "0")), int(d.get("oct", 0)))
                    return self._send(200, {"ok": True})
                if pth.startswith("/assign"):
                    i = st.assign_slot(int(d["slot"]), str(d.get("digit", "6")),
                                       int(d.get("oct", 0)))
                    return self._send(200, {"ok": i is not None, "id": i})
                if pth.startswith("/keychange_del"):
                    return self._send(200, {"ok": st.del_key_change(int(d.get("i", -1)))})
                if pth.startswith("/keychange"):
                    ok = st.add_key_change(float(d["x"]), float(d["y"]), str(d.get("key", "C")))
                    return self._send(200, {"ok": ok})
                if pth.startswith("/target"):
                    return self._send(200, {"ok": st.set_target(str(d.get("key", "C")))})
                if pth.startswith("/size_row"):
                    n = st.set_size_row(int(d["row"]), str(d.get("cls", "main")))
                    return self._send(200, {"ok": True, "n": n})
                if pth.startswith("/size"):
                    n = st.set_size(d.get("ids") or [], str(d.get("cls", "main")))
                    return self._send(200, {"ok": True, "n": n})
                if pth.startswith("/undo"):
                    return self._send(200, {"ok": st.undo()})
                self._send(404, {"ok": False})

    return H


def main(argv=None):
    import jp.library as LIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--song", default="", help="（旧参数）直接打开 SONGS 里的某首")
    ap.add_argument("--lib", default="", help="直接打开库里某条 id")
    a = ap.parse_args(argv)

    st = None
    if a.song and a.song in SONGS:
        orig, ocr_path, key_name = SONGS[a.song]
        st = State(orig, ocr_path, key_name, title=a.song)
    elif a.lib:
        items = LIB.load()
        it = items.get(a.lib)
        if it:
            tp = it.get("ocr_txt") or LIB.ocr_path_for_id(it["id"])
            if os.path.exists(tp):
                key = it.get("key") or LIB.detect_key(open(tp, encoding="utf8").read()) or "C"
                print("正在打开乐谱库里的: %s（约 10–30 秒）" % it.get("song"), flush=True)
                st = State(it["image"], tp, key, title=it.get("title") or it.get("song") or "",
                   cache_key=a.lib,
                   strength=int((it or {}).get("strength") or 1))
                ACTIVE["lib_id"] = a.lib
                if it.get("target"):
                    pc = M.parse_key(it["target"])
                    if pc is not None:
                        st.target_pc = int(pc)
    # 默认**不**自动打开：打开一页要 10–30 秒，启动就该立刻能用乐谱库
    with ALOCK:
        ACTIVE["st"] = st

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler())
    print("乐谱库:  http://127.0.0.1:%d/lib" % a.port, flush=True)
    print("纠错页:  http://127.0.0.1:%d/" % a.port, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
