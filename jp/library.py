"""乐谱库：导入、归档、批量处理。

一张原始谱面 = 一条记录：
    {id, song, image, ocr_txt, key, target, status, out, notes, uncovered, updated}

`status`：new（只导入了图片）/ ocr（OCR 文本就绪）/ done（已出红字图）/ error

处理流程与 `jp.batch` 完全一致：
    图片 --(Xiaomi OCR，缓存到 out\\batch\\_ocr)--> 文本
         --> merge_page（几何 + 文本对齐）--> render_fixed_image（红字）--> out\\batch\\<歌>\\<页>_fixed.jpg

OCR 结果按 `歌名__页名.txt` 缓存，重跑不再烧 GPU。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

# 同一个进程里 HTTP 是多线程的，读-改-写必须串行，否则 library.json 会被写坏/占用
_WLOCK = threading.RLock()
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import core
from . import emit_fixed as EF
from . import merge_octave as MO
from . import music as M
from . import replace_img as RI
from . import xiaomi_ocr as XO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(ROOT, "library.json")
# 算法版本：改了几何/对齐/渲染逻辑就 +1，所有打开缓存自动失效
PIPE_VER = 3

LIB_DIR = os.path.join(ROOT, "library")          # 自己的库：图片/OCR/缓存都复制进来
BATCH = os.path.join(ROOT, "out", "batch")
OCR_DIR = os.path.join(BATCH, "_ocr")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff",
           ".avif", ".heic", ".heif")

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def store_dir(i: str) -> str:
    d = os.path.join(LIB_DIR, i)
    os.makedirs(d, exist_ok=True)
    return d


def store_image(i: str, src: str) -> str:
    """把图片**复制**进自己的库，之后就不依赖原文件了。"""
    ext = os.path.splitext(src)[1].lower() or ".jpg"
    dst = os.path.join(store_dir(i), "page" + ext)
    if os.path.abspath(src) != os.path.abspath(dst):
        import shutil
        shutil.copyfile(src, dst)
    return dst


def store_ocr_text(i: str, text: str) -> str:
    dst = os.path.join(store_dir(i), "ocr.txt")
    with open(dst, "w", encoding="utf8") as f:
        f.write(text or "")
    return dst


def cache_path(i: str) -> str:
    return os.path.join(store_dir(i), "cache.pkl")


def src_sig(path: str):
    """来源图指纹：大小 + 前 1MB 内容哈希。

    不用 mtime：归档（复制进 library\\）或换目录会把 mtime 改掉，
    但**内容没变**，缓存应该继续有效。内容真的变了指纹就变，缓存自动作废。
    """
    try:
        st = os.stat(path)
        h = hashlib.sha1()
        with open(path, "rb") as f:
            h.update(f.read(1 << 20))
        return [int(st.st_size), h.hexdigest()[:16]]
    except Exception:
        return None


def save_cache(i: str, payload: dict) -> bool:
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["pipe"] = PIPE_VER
    import pickle
    try:
        p = cache_path(i)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        os.replace(tmp, p)
        return True
    except Exception:
        return False


def ocr_sig(path: str) -> str:
    """OCR 文本指纹：文本一改，打开缓存就作废（否则会拿旧的错结果渲染）。"""
    try:
        h = hashlib.sha1()
        h.update(open(path, "rb").read())
        return h.hexdigest()[:16]
    except Exception:
        return ""


def load_cache(i: str, image: str, ocr_sig_expected: str = ""):
    """读缓存；来源图内容变了就作废（mtime 变化不算）。"""
    import pickle
    p = cache_path(i)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
        if int(d.get("pipe", 0)) != PIPE_VER:
            return None              # 算法升级过 → 旧结果不再可信
        sig = src_sig(image)
        old = d.get("src_sig")
        if ocr_sig_expected and d.get("ocr_sig") != ocr_sig_expected:
            return None          # OCR 文本变了 → 旧缓存作废
        if old == sig:
            return d
        # 兼容旧版缓存（[名字, 大小, mtime]）：大小一致就先用着
        if isinstance(old, list) and len(old) == 3 and sig and old[1] == sig[0]:
            d["src_sig"] = sig
            return d
        return None
    except Exception:
        return None


# ---------------------------------------------------------------- 导入源管理
def sources() -> List[dict]:
    return _read_all()["sources"]


def add_source(name: str, kind: str, note: str = "") -> dict:
    d = _read_all()
    sid = hashlib.sha1((name + kind + str(time.time_ns())).encode("utf8")).hexdigest()[:8]
    ent = dict(id=sid, name=name, kind=kind, note=note,
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    d["sources"].append(ent)
    _write_all(d)
    return ent


def del_source(sid: str) -> bool:
    d = _read_all()
    d["sources"] = [s for s in d["sources"] if s.get("id") != sid]
    _write_all(d)
    return True


def import_uploaded(filename: str, data: bytes, song: str = "", source_id: str = "",
                    key: str = "") -> Tuple[str, bool]:
    """把浏览器上传的一个文件收进库里（复制进 library/<id>/，之后不依赖原文件）。

    支持 jpg/png/gif/bmp/webp/tif/avif 等：cv2 能解的原样存，
    解不了的（avif 走 Pillow）转成 jpg；heic 需要额外插件，会明确报错。
    """
    h = hashlib.sha1(data)
    h.update(os.path.basename(filename).encode("utf8"))
    i = h.hexdigest()[:12]
    items = load()
    if i in items:
        return i, False
    song = song or os.path.splitext(os.path.basename(filename))[0]
    dst = store_image_any(i, data, filename)
    _now = int(time.time())
    it = dict(id=i, song=song, image=dst, origin=filename, title="", order=len(items),
              t0=_now, t1=_now,
              ocr_txt=ocr_path_for_id(i),
              key=key, target="C", status="new",
              out=out_path_for(song, dst), notes=0, uncovered=0,
              keys=[], multi=False, dup=False, source_id=source_id,
              updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    it["dup"] = bool(looks_transposed(dst))
    items[i] = it
    _save_all(dict(items=items))
    return i, True


def archive_existing(i: str) -> Optional[dict]:
    """把「还在引用外部文件」的旧记录复制进库目录。"""
    items = load()
    it = items.get(i)
    if not it:
        return None
    src = it.get("image", "")
    if os.path.exists(src) and os.path.abspath(os.path.dirname(src)) != os.path.abspath(store_dir(i)):
        it["origin"] = it.get("origin") or src
        it["image"] = store_image(i, src)
        if it.get("ocr_txt") and os.path.exists(it["ocr_txt"]):
            txt = open(it["ocr_txt"], encoding="utf8").read()
            it["ocr_txt"] = store_ocr_text(i, txt)
        it["out"] = out_path_for(it["song"], it["image"])
        it["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        items[i] = it
        _save_all(dict(items=items))
    return it


def safe(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\r\n\t]', "_", name or "")
    return s.strip(" .")[:120] or "unnamed"


def _id(path: str) -> str:
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf8")).hexdigest()[:12]


# ---------------------------------------------------------------- 图片解码
def decode_bgr(data_or_path, is_path: bool = True):
    """cv2 优先；cv2 解不了的（avif 等）退到 PIL。都解不了返回 None。"""
    import cv2
    try:
        if is_path:
            raw = np.fromfile(data_or_path, np.uint8)
        else:
            raw = np.frombuffer(data_or_path, np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
    except Exception:
        pass
    try:
        import io as _io
        from PIL import Image
        src = data_or_path if is_path else _io.BytesIO(data_or_path)
        im = Image.open(src)
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        arr = np.array(im)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr[:, :, ::-1].copy()          # RGB -> BGR
    except Exception:
        return None


def store_image_any(i: str, data: bytes, filename: str) -> str:
    """把上传的图片收进库：cv2 能解就原样存，解不了（avif 之类）就转成 jpg。"""
    import cv2
    ext = os.path.splitext(filename)[1].lower()
    raw = np.frombuffer(data, np.uint8)
    ok_orig = False
    try:
        ok_orig = cv2.imdecode(raw, cv2.IMREAD_COLOR) is not None
    except Exception:
        ok_orig = False
    if ok_orig and ext in IMG_EXT:
        dst = os.path.join(store_dir(i), "page" + ext)
        with open(dst, "wb") as f:
            f.write(data)
        return dst
    bgr = decode_bgr(data, is_path=False)
    if bgr is None:
        raise RuntimeError("这个格式解不开：%s" % (ext or "未知"))
    dst = os.path.join(store_dir(i), "page.jpg")
    cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(dst)
    return dst


# ---------------------------------------------------------------- 多调识别
KEY_RE = re.compile(r"1\s*=\s*([♭♯＃ｂb#]?[A-G][#b]?)")


def detect_keys(txt: str) -> List[dict]:
    """从 OCR 文本里找出**所有** `1=X`（可能中途转调多次）。

    返回 [{key: 'G', line: 0}, {key: 'D', line: 37}]，line 是 OCR 文本里的行号，
    后面按"行号 / 总行数"折算成页内相对高度，作为中途转调的位置。
    """
    out: List[dict] = []
    lines = (txt or "").splitlines()
    for i, ln in enumerate(lines):
        for m in KEY_RE.finditer(ln):
            pc = M.parse_key(m.group(1))
            if pc is None:
                continue
            nm = KEY_NAMES[pc % 12]
            if out and out[-1]["key"] == nm:      # 相邻重复（谱头常重复写）不记
                continue
            out.append(dict(key=nm, line=i))
    return out[:8]


# ---------------------------------------------------------------- 已转调检测
def red_metrics(path: str) -> Optional[dict]:
    import cv2
    try:
        buf = np.fromfile(path, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        B, G, R = [bgr[:, :, i].astype(np.int32) for i in range(3)]
        strong = (R > 140) & (G < 70) & (B < 70)      # 接近纯红 = 我们/easytr 画的红字
        ink = (R + G + B) < 420
        ni = max(1, int(ink.sum()))
        return dict(ratio=float(strong.sum()) / ni, n=int(strong.sum()))
    except Exception:
        return None


def looks_transposed(path: str) -> bool:
    """判断这张图是不是**已经转调过的输出**（红字占比高）。

    标定（语料 50 原始 + 27 转调输出）：原始页 pure-red 占比 ≤ 0.28（最高那张
    其实是转调图），转调输出 ≥ 0.17。取 0.06 + 绝对像素 > 3000 作判据，
    实测 0 漏判。
    """
    m = red_metrics(path)
    if m is None:
        return False
    return m["ratio"] > 0.06 and m["n"] > 3000


def refresh_meta(i: str) -> Optional[dict]:
    """按磁盘现状刷新一条记录：多调、是否已转调。"""
    items = load()
    it = items.get(i)
    if not it:
        return None
    tp = it.get("ocr_txt") or ocr_path_for(it["song"], it["image"])
    if os.path.exists(tp):
        keys = detect_keys(open(tp, encoding="utf8").read())
        if keys:
            it["keys"] = keys
            if not it.get("key"):
                it["key"] = keys[0]["key"]
    if os.path.exists(it.get("image", "")):
        it["dup"] = bool(looks_transposed(it["image"]))
    items[i] = it
    save(items)
    return it


def _read_all() -> dict:
    with _WLOCK:
        return _read_all_nolock()


def _read_all_nolock() -> dict:
    if not os.path.exists(LIB_PATH):
        return dict(items={}, sources=[])
    try:
        d = json.load(open(LIB_PATH, encoding="utf8"))
        d.setdefault("items", {})
        d.setdefault("sources", [])
        return d
    except Exception:
        return dict(items={}, sources=[])


def _write_all(d: dict) -> None:
    """写 library.json：Windows 上 os.replace 遇到目标被占用会抛 WinError 5，
    所以带锁 + 重试 + 直接覆盖兜底（之前重扫时就因为并发写直接抛异常了）。"""
    d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data = json.dumps(d, ensure_ascii=False, indent=1)
    with _WLOCK:
        for attempt in range(6):
            tmp = "%s.tmp%d" % (LIB_PATH, attempt)
            try:
                with open(tmp, "w", encoding="utf8") as f:
                    f.write(data)
                os.replace(tmp, LIB_PATH)
                return
            except (PermissionError, OSError):
                time.sleep(0.12 * (attempt + 1))
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        with open(LIB_PATH, "w", encoding="utf8") as f:      # 兜底：直接覆盖
            f.write(data)


def load() -> Dict[str, dict]:
    return _read_all()["items"]


def save(items: Dict[str, dict]) -> None:
    d = _read_all()
    d["items"] = items
    _write_all(d)


def _save_all(d: dict) -> None:
    """只覆盖 items，**保留 sources**（以前这里直接把 sources 冲掉了）。"""
    cur = _read_all()
    if "items" in d:
        cur["items"] = d["items"]
    for k in ("sources",):
        if k in d:
            cur[k] = d[k]
    _write_all(cur)


def ocr_path_for_id(i: str) -> str:
    """每页自己的 OCR 文本：library/<id>/ocr.txt。

    以前按「歌名__文件名」放在 out\\batch\\_ocr 里，可 avif 转码后文件名统一变成
    page.jpg，多页撞成同一个缓存文件 —— 后一页的 OCR 覆盖前一页，
    于是十几页都读到了同一份歌词（"光与影的对白"就是这么来的）。"""
    return os.path.join(store_dir(i), "ocr.txt")


def ocr_path_for(song: str, image: str, i: str = "") -> str:
    """兼容旧调用；带 id 时永远按 id 走，不带则用图片路径哈希保证唯一。"""
    if i:
        return ocr_path_for_id(i)
    h = hashlib.sha1(os.path.abspath(image).encode("utf8")).hexdigest()[:10]
    return os.path.join(OCR_DIR, safe(song) + "__" + h + ".txt")


def out_path_for(song: str, image: str) -> str:
    stem = safe(os.path.splitext(os.path.basename(image))[0])
    return os.path.join(BATCH, safe(song), stem + "_fixed.jpg")


def scan(root: str, key: str = "") -> Tuple[int, int, int]:
    """扫目录（或单个文件）把图片登记进库里；已存在的只更新路径，不动已有状态。"""
    items = load()
    added = seen = 0
    root = os.path.abspath(root)
    files_only: List[str] = []
    if os.path.isfile(root):
        files_only = [root]
        root_dir = os.path.dirname(root)
        pairs = [(root_dir, [], [os.path.basename(root)])]
    else:
        pairs = list(os.walk(root))
    for dirpath, _dirs, files in pairs:
        song = os.path.basename(dirpath)
        if os.path.normcase(dirpath) == os.path.normcase(root):
            song = os.path.basename(dirpath)
        if _is_inside(dirpath, LIB_DIR):        # 绝不扫自己的库目录（会扫出缩略图）
            continue
        for f in sorted(files):
            if not f.lower().endswith(IMG_EXT):
                continue
            if f.startswith("thumb") or f.startswith("preview"):
                continue
            if "(原调" in f:                     # 排除对照图
                continue
            path = os.path.join(dirpath, f)
            i = _id(path)
            seen += 1
            ext = os.path.splitext(f)[1].lower()
            src_path = path
            if ext in (".avif", ".heic", ".heif") and i not in items:
                # 下游（core/兼容层）读不了这些格式 → 直接转码存进库
                try:
                    with open(path, "rb") as fh:
                        path = store_image_any(i, fh.read(), f)
                    ext = os.path.splitext(path)[1].lower()
                except Exception:
                    continue
            if i in items:
                items[i]["image"] = items[i].get("image") or path
                items[i]["origin"] = items[i].get("origin") or path
                items[i]["song"] = items[i].get("song") or song
                continue
            _t0, _t1 = _stats_pair(path)
            items[i] = dict(id=i, song=song, image=path, title="", order=seen,
                            origin=src_path, t0=_t0, t1=_t1,
                            ocr_txt=ocr_path_for_id(i),
                            key=key, target="C", status="new",
                            out=out_path_for(song, path),
                            notes=0, uncovered=0, keys=[], multi=False, dup=False,
                            updated=time.strftime("%Y-%m-%d %H:%M:%S"))
            if os.path.exists(items[i]["ocr_txt"]):
                items[i]["status"] = "ocr"
                _ks = detect_keys(open(items[i]["ocr_txt"], encoding="utf8").read())
                if _ks:
                    items[i]["keys"] = _ks
                    items[i]["key"] = items[i].get("key") or _ks[0]["key"]
                    items[i]["multi"] = len(_ks) > 1
            if os.path.exists(items[i]["out"]):
                items[i]["status"] = "done"
            items[i]["dup"] = bool(looks_transposed(path))
            added += 1
    save(items)
    return added, seen, len(items)


def _stats_pair(path: str):
    """来源文件的 (创建时间, 修改时间)，秒。"""
    try:
        st = os.stat(path)
        return int(st.st_ctime), int(st.st_mtime)
    except Exception:
        return None, None


def backfill_times() -> int:
    """老记录补上时间戳（优先用 origin 原文件的时间）。"""
    items = load()
    n = 0
    for it in items.values():
        if it.get("t0"):
            continue
        src = it.get("origin") if it.get("origin") and os.path.exists(it.get("origin") or "") \
            else it.get("image")
        c, m = _stats_pair(src or "")
        if c:
            it["t0"], it["t1"] = c, m
            n += 1
    if n:
        save(items)
    return n


# ---------------------------------------------------------------- 自动按份分组
def _bigrams(s: str):
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _sim(a, b) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(max(1, len(a | b)))


def auto_group(ids=None, log=print) -> dict:
    """把页面按「同一份简谱」分组。

    判据（从强到弱）：
    1. 同一个目录（子文件夹名）→ 几乎肯定同一份；
    2. 歌词/谱面文字相似（字符二元组 Jaccard ≥ 0.25）→ 同一份（跨目录也认）；
    3. 都在同一目录且没有文字时，看**文件时间间隔**：按顺序算相邻间隔，
       用 3×中位数 当切分阈值（你的例子：谱内 5s、谱间 20s，阈值≈15s），
       超过阈值就断开。
    4. 时间缺失的页不单独断开，跟在后一页那一组里（总比乱拆好）。
    """
    backfill_times()
    items = load()
    seq = [it for it in items.values() if (ids is None or it["id"] in set(ids))]
    if not seq:
        return dict(groups=0, pages=0)
    for it in seq:
        tp = it.get("ocr_txt") or ocr_path_for(it["song"], it["image"])
        txt = ""
        if tp and os.path.exists(tp):
            try:
                txt = open(tp, encoding="utf8").read()
            except Exception:
                txt = ""
        it["_bg"] = _bigrams(txt)
    # 目录内按时间排，目录之间不混
    seq.sort(key=lambda x: ((x.get("origin") or x.get("image") or "").rsplit("\\", 1)[0].lower(),
                            x.get("t0") or 0, x.get("order") or 0))
    # 目录内的切分阈值：3× 正间隔中位数（至少 10s）
    gaps_by_dir = {}
    prev = None
    for it in seq:
        d = (it.get("origin") or it.get("image") or "").rsplit("\\", 1)[0].lower()
        t = it.get("t0") or 0
        if prev and prev[0] == d and t and prev[1]:
            gaps_by_dir.setdefault(d, []).append(abs(t - prev[1]))
        prev = (d, t if t else (prev[1] if prev and prev[0] == d else 0))
    thr = {}
    stats_by_dir = {}
    for d, gs in gaps_by_dir.items():
        # 只用**正间隔**；均值+标准差（需求方要求），σ 用总体标准差
        pos = [float(g) for g in gs if g > 0]
        if not pos:
            thr[d] = 10.0
            stats_by_dir[d] = (0.0, 0.0)
            continue
        mu = sum(pos) / len(pos)
        sd = (sum((x - mu) ** 2 for x in pos) / len(pos)) ** 0.5
        stats_by_dir[d] = (mu, sd)
        # 谱内间隔通常几分钟内；谱间是"下载完一首再下一首"的长间隔。
        # 阈值 = 均值 + 1.5σ（至少 10s，避免全是 0/1 秒时乱切）
        thr[d] = max(10.0, mu + 1.5 * sd)

    groups = []
    cur = []
    for it in seq:
        if not cur:
            cur = [it]
            continue
        last = cur[-1]
        d1 = (last.get("origin") or last.get("image") or "").rsplit("\\", 1)[0].lower()
        d2 = (it.get("origin") or it.get("image") or "").rsplit("\\", 1)[0].lower()
        same_dir = d1 == d2
        sim = _sim(last.get("_bg"), it.get("_bg"))
        t1, t2 = last.get("t0") or 0, it.get("t0") or 0
        gap = abs(t2 - t1) if (t1 and t2) else None
        # 断开条件：不同目录 + 文本不像；或者同目录但时间间隔远超阈值且文本也不像
        brk = False
        if not same_dir and sim < 0.25:
            brk = True
        elif gap is not None and gap > thr.get(d2, 10.0) and sim < 0.25:
            brk = True
        if brk:
            groups.append(cur)
            cur = [it]
        else:
            cur.append(it)
    if cur:
        groups.append(cur)

    n_page = 0
    made: Dict[str, str] = {}
    for gi, g in enumerate(groups):
        title = ""
        cnt = {}
        for it in g:
            n_page += 1
            t = (it.get("title") or "").strip()
            if t:
                cnt[t] = cnt.get(t, 0) + 1
        if cnt:
            title = max(cnt.items(), key=lambda kv: kv[1])[0]
        if not title:
            title = g[0].get("song") or ""
        gid = "g_" + (g[0].get("id") or str(gi))
        made[gid] = title
        for pos, it in enumerate(g):
            it.pop("_bg", None)
            items[it["id"]]["group"] = gid
            items[it["id"]]["group_title"] = title
            items[it["id"]]["group_pos"] = pos + 1
            items[it["id"]]["group_n"] = len(g)
            items[it["id"]]["order"] = pos
    save(items)
    d = _read_all()
    gs = d.setdefault("groups", {})
    for gid, title in made.items():
        old = gs.get(gid) or {}
        gs[gid] = dict(id=gid, name=old.get("name") or title,
                       name_locked=old.get("name_locked", True),
                       created=old.get("created") or time.strftime("%Y-%m-%d %H:%M:%S"))
    d["groups"] = gs
    _write_all(d)
    log("分成 %d 份，共 %d 页" % (len(groups), n_page))
    return dict(groups=len(groups), pages=n_page,
                detail=[dict(title=g[0].get("group_title") or g[0].get("song"),
                             n=len(g)) for g in groups[:40]])


def _common_prefix(dirs: List[str]) -> str:
    a = [os.path.abspath(d) for d in dirs]
    if not a:
        return ""
    parts0 = a[0].split(os.sep)
    others = [x.split(os.sep) for x in a]
    common = []
    for i, seg in enumerate(parts0):
        if all(i < len(p) and p[i].lower() == seg.lower() for p in others):
            common.append(seg)
        else:
            break
    return os.sep.join(common)


def _is_inside(child: str, parent: str) -> bool:
    c = os.path.normcase(os.path.abspath(child)).rstrip("\\/") + os.sep
    p = os.path.normcase(os.path.abspath(parent)).rstrip("\\/") + os.sep
    return c.startswith(p)


def _flatten_paths(paths: List[str]) -> List[str]:
    """去掉被别的目录包含的目录（扫描本来就是递归的，留着只会乱）。"""
    out: List[str] = []
    for p in sorted(set(paths), key=len):
        if any(_is_inside(p, q) for q in out):
            continue
        out.append(p)
    return out


def collapse_sources() -> int:
    """把「按路径登记」的目录折叠成尽量少的顶层目录。

    需求方：**要读子文件夹，但别把子文件夹都列出来**。扫描本身是递归的，
    所以把 51 个歌曲目录改成一个共同根目录即可，仍然覆盖全部子目录。
    """
    d = _read_all()
    ss = d.get("sources") or []
    up = [s for s in ss if s.get("kind") != "path"]
    paths = [os.path.abspath(s.get("note") or "") for s in ss
             if s.get("kind") == "path" and s.get("note")]
    if not paths:
        if len(ss) != len(up):
            d["sources"] = up
            _write_all(d)
        return len(up)
    groups: Dict[str, List[str]] = {}
    for p in paths:
        drv = os.path.splitdrive(p)[0].lower() or "?"
        groups.setdefault(drv, []).append(p)
    keep: List[str] = []
    for _drv, ds in groups.items():
        pref = _common_prefix(ds)
        depth = len([x for x in pref.split(os.sep) if x])
        keep.extend([pref] if depth >= 2 else ds)      # 别退化成盘符根
    keep = [x for x in _flatten_paths(keep) if not _is_inside(x, LIB_DIR)]

    old = {os.path.abspath(s.get("note") or ""): s for s in ss if s.get("kind") == "path"}
    new_ss = list(up)
    for p in keep:
        base = old.get(p)
        new_ss.append(base or dict(
            id=hashlib.sha1(("path" + p).encode("utf8")).hexdigest()[:8],
            name=os.path.basename(p.rstrip("\\/")) or p, kind="path", note=p,
            time=time.strftime("%Y-%m-%d %H:%M:%S")))
    d["sources"] = new_ss
    _write_all(d)
    return len(new_ss)


def ensure_preview(i: str, width: int = 1100) -> Optional[str]:
    """悬停放大用的中等图（宽 1100），比缩略图清晰、比原图小。"""
    it = get(i)
    if not it:
        return None
    dst = os.path.join(store_dir(i), "preview.jpg")
    src = it.get("image") or ""
    try:
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
        bgr = decode_bgr(src)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        scale = min(1.0, width / float(max(1, w)))
        import cv2
        small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
        cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 86])[1].tofile(dst)
        return dst
    except Exception:
        return None


def thumb_path(i: str) -> str:
    return os.path.join(store_dir(i), "thumb.jpg")


def ensure_thumb(i: str) -> Optional[str]:
    """生成小缩略图（宽 300），供两栏工作区悬停预览用。失败返回 None。"""
    it = get(i)
    if not it:
        return None
    tp = thumb_path(i)
    src = it.get("image") or ""
    try:
        if os.path.exists(tp) and os.path.getmtime(tp) >= os.path.getmtime(src):
            return tp
        bgr = decode_bgr(src)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        scale = 300.0 / max(1, w)
        small = cv2_resize(bgr, scale)
        import cv2
        cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])[1].tofile(tp)
        return tp
    except Exception:
        return None


def cv2_resize(bgr, scale: float):
    import cv2
    h, w = bgr.shape[:2]
    if scale >= 1.0:
        return bgr
    return cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 分组（谱）管理
def groups() -> Dict[str, dict]:
    return _read_all().get("groups") or {}


def reorder_pages(gid: str, ids: List[str]) -> int:
    """组内拖动排序：把该组的页面顺序写成 group_pos（gid 为空 = 未分类那栏）。"""
    d = _read_all()
    g = (d.get("groups") or {}).get(gid) or {}
    n = 0
    for pos, i in enumerate(ids):
        it = d["items"].get(i)
        if not it:
            continue
        it["group"] = gid or ""
        it["group_title"] = g.get("name", "") if gid else ""
        it["group_pos"] = pos + 1 if gid else pos
        it["group_n"] = len(ids) if gid else 0
        it["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        n += 1
    _write_all(d)
    return n


def group_reorder(ids: List[str]) -> int:
    """手动排列左侧「份」的先后顺序。"""
    d = _read_all()
    gs = d.setdefault("groups", {})
    n = 0
    for pos, gid in enumerate(ids):
        g = gs.get(gid)
        if not g:
            continue
        g["order"] = pos
        n += 1
    d["groups"] = gs
    _write_all(d)
    return n


def clear_titles(ids=None, group: str = "", all_pages: bool = False) -> int:
    """清空标题：可以按 id 列表、按组、或全库。

    需求方：**加入编组不代表就要被命名**——AI 认错时得能一键抹掉，
    否则错标题就一直挂在页上，分组也跟着错。
    """
    d = _read_all()
    n = 0
    for it in d["items"].values():
        if all_pages:
            pass
        elif group:
            if it.get("group") != group:
                continue
        elif ids is not None:
            if it["id"] not in set(ids):
                continue
        else:
            continue
        if it.get("title"):
            it["title"] = ""
            n += 1
    _write_all(d)
    return n


def group_new(name: str = "") -> dict:
    d = _read_all()
    gs = d.setdefault("groups", {})
    name = (name or "").strip() or ("新谱 %d" % (len(gs) + 1))
    gid = "g_" + hashlib.sha1((name + str(time.time_ns())).encode("utf8")).hexdigest()[:8]
    gs[gid] = dict(id=gid, name=name, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    d["groups"] = gs
    _write_all(d)
    return gs[gid]


def group_set(gid: str, **kw) -> Optional[dict]:
    d = _read_all()
    gs = d.setdefault("groups", {})
    g = gs.get(gid)
    if not g:
        return None
    for k in ("name", "collapsed"):
        if k in kw:
            g[k] = kw[k]
    _write_all(d)
    return g


def group_del(gid: str) -> int:
    """删掉一份谱（组），里面的页退回未分类。"""
    d = _read_all()
    d.setdefault("groups", {}).pop(gid, None)
    n = 0
    for it in d["items"].values():
        if it.get("group") == gid:
            it["group"] = ""
            it["group_title"] = ""
            it["group_pos"] = 0
            it["group_n"] = 0
            n += 1
    _write_all(d)
    return n


def move_to(ids: List[str], gid: str) -> int:
    """把页移进某一组（gid 为空 = 退回未分类），顺序接在组尾。"""
    d = _read_all()
    gs = d.setdefault("groups", {})
    name = (gs.get(gid) or {}).get("name", "") if gid else ""
    nxt = 0
    if gid:
        nxt = 1 + max([int(it.get("group_pos") or 0) for it in d["items"].values()
                       if it.get("group") == gid] or [0])
    n = 0
    for i in ids:
        it = d["items"].get(i)
        if not it:
            continue
        it["group"] = gid or ""
        it["group_title"] = name
        it["group_pos"] = nxt if gid else 0
        nxt += 1
        it["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        n += 1
    # 修正各组的页数
    cnt: Dict[str, int] = {}
    for it in d["items"].values():
        g = it.get("group")
        if g:
            cnt[g] = cnt.get(g, 0) + 1
    for it in d["items"].values():
        g = it.get("group")
        it["group_n"] = cnt.get(g, 0) if g else 0
    _write_all(d)
    return n


# ---------------------------------------------------------------- 垃圾箱 / 重复检测
def soft_delete(ids: List[str]) -> int:
    """软删除：标记 deleted，**永不再出现**（扫描/导入都不会把它复活）。"""
    d = _read_all()
    n = 0
    for i in ids:
        it = d["items"].get(i)
        if not it or it.get("deleted"):
            continue
        it["deleted"] = True
        it["deleted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        it["updated"] = it["deleted_at"]
        n += 1
    _write_all(d)
    return n


def restore(ids: List[str]) -> int:
    d = _read_all()
    n = 0
    for i in ids:
        it = d["items"].get(i)
        if not it or not it.get("deleted"):
            continue
        it["deleted"] = False
        it["deleted_at"] = ""
        n += 1
    _write_all(d)
    return n


def purge(ids: List[str]) -> int:
    """永久删除：连文件和记录一起删掉（垃圾箱里"彻底清掉"才用）。"""
    import shutil
    d = _read_all()
    n = 0
    for i in ids:
        it = d["items"].pop(i, None)
        if not it:
            continue
        shutil.rmtree(os.path.join(LIB_DIR, i), ignore_errors=True)
        n += 1
    _write_all(d)
    return n


def trash() -> List[dict]:
    return [it for it in load().values() if it.get("deleted")]


def _phash_of(it: dict) -> Optional[str]:
    """感知哈希（dhash 64 位）：格式/尺寸不同但同一张图也能认出来。"""
    if it.get("phash"):
        return it["phash"]
    try:
        import cv2
        bgr = decode_bgr(it.get("image") or "")
        if bgr is None:
            return None
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        v = 0
        for b in bits:
            v = (v << 1) | int(bool(b))
        return "%016x" % v
    except Exception:
        return None


def _hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 64


def mark_dupes(log=print) -> dict:
    """给疑似重复的页打标（只标记，不删）。"""
    items = load()
    ph: Dict[str, str] = {}
    for i, it in items.items():
        if it.get("deleted"):
            continue
        h = _phash_of(it)
        if not h:
            continue
        it["phash"] = h
        it["dup_of"] = ""
        ph[i] = h
    ids = list(ph)
    n = 0
    seen: List[str] = []
    for a in ids:
        for b in seen:
            if _hamming(ph[a], ph[b]) <= 6:
                items[a]["dup_of"] = b
                n += 1
                break
        else:
            seen.append(a)
    _save_all(dict(items=items))
    log("疑似重复 %d 页（共检查 %d 页）" % (n, len(ids)))
    return dict(dupes=n, checked=len(ids))


def _norm_title(t: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKC", t or "").lower()
    t = re.sub(r"[（(【\[].*?[）)】\]]", "", t)          # 去掉括号补充（动画片名之类）
    t = re.sub(r"[\s\-_·・.。,，:：!！?？'\"“”‘’/\\|]+", "", t)
    return t


def auto_group_by_title(ids=None, log=print) -> dict:
    """按 **AI 标题** 重新分份（比时间可靠得多）。

    1. 先按规范化标题聚类：二元组 Jaccard ≥ 0.55 或互相包含 → 同一份；
    2. 没有标题的页：同目录 + 与某簇时间间隔 ≤ 均值+1.5σ → 并入该簇，否则单开一组
       （组名用目录名 + 序号）；
    3. 组名锁定（name_locked），之后「用 AI 标题汇总谱名」不会覆盖它。
    """
    backfill_times()
    items = load()
    seq = [it for it in items.values() if (ids is None or it["id"] in set(ids))]
    if not seq:
        return dict(groups=0, pages=0)
    seq.sort(key=lambda x: (x.get("t0") or 0, x.get("order") or 0))
    for it in seq:
        it["_t"] = _norm_title(it.get("title") or "")
        it["_bg"] = _bigrams(it["_t"])

    clusters = []      # [{key, bg, name, items:[...]}]
    for it in seq:
        t, bg = it["_t"], it["_bg"]
        if not t:
            continue
        hit = None
        for c in clusters:
            if c["key"] == t or _sim(c["bg"], bg) >= 0.55 or (
                    len(t) >= 2 and (t in c["key"] or c["key"] in t)):
                hit = c
                break
        if hit is None:
            clusters.append(dict(key=t, bg=set(bg), name=(it.get("title") or t), items=[]))
            hit = clusters[-1]
        hit["items"].append(it)

    # 计算每个目录的间隔阈值（均值+1.5σ）
    by_dir = {}
    for it in seq:
        d = os.path.dirname(it.get("origin") or it.get("image") or "").lower()
        by_dir.setdefault(d, []).append(it.get("t0") or 0)
    thr = {}
    for d, ts in by_dir.items():
        ts = sorted(x for x in ts if x)
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        if not gaps:
            thr[d] = 10.0
            continue
        mu = sum(gaps) / len(gaps)
        sd = (sum((g - mu) ** 2 for g in gaps) / len(gaps)) ** 0.5
        thr[d] = max(10.0, mu + 1.5 * sd)

    # 没标题的页：**不再硬塞进某一组**，留在未分类（右侧），由人来决定归哪一份
    untitled = [it for it in seq if not it["_t"]]

    if not clusters and not untitled:
        return dict(groups=0, pages=0)

    d = _read_all()
    old_gs = d.get("groups") or {}
    by_name = {}
    for gid, g in old_gs.items():
        by_name.setdefault(_norm_title(g.get("name") or ""), gid)

    gs = {}
    for c in clusters:
        key = _norm_title(c["name"])
        gid = by_name.get(key) or "g_" + hashlib.sha1(("t:" + key).encode("utf8")).hexdigest()[:8]
        gs[gid] = dict(id=gid, name=c["name"], name_locked=True,
                       order=len(gs),
                       created=(old_gs.get(gid) or {}).get("created")
                       or time.strftime("%Y-%m-%d %H:%M:%S"))
        for pos, it in enumerate(sorted(c["items"], key=lambda x: x.get("t0") or 0)):
            node = d["items"][it["id"]]
            node["group"] = gid
            node["group_title"] = c["name"]
            node["group_pos"] = pos + 1
            node["group_n"] = len(c["items"])
        c["gid"] = gid

    # 没标题的页 → 未分类；原来抓着它们的旧组会被清空（下面统一删空组）
    for it in untitled:
        node = d["items"][it["id"]]
        node["group"] = ""
        node["group_title"] = ""
        node["group_pos"] = 0
        node["group_n"] = 0
    for it in d["items"].values():
        g = it.get("group")
        if g and g not in gs:
            it["group"] = ""
            it["group_title"] = ""
            it["group_n"] = 0
    d["groups"] = gs
    _write_all(d)
    npage = sum(len(c["items"]) for c in clusters)
    log("按 AI 标题分成 %d 份，共 %d 页；未识别 %d 页放在右侧未分类"
        % (len(clusters), npage, len(untitled)))
    return dict(groups=len(clusters), pages=npage, untitled=len(untitled),
                detail=[dict(title=c["name"], n=len(c["items"])) for c in clusters[:80]])


def aggregate_group_titles(log=print) -> dict:
    """把 AI 猜出来的页标题按组汇总：每组取出现次数最多的标题当组名。

    逐页猜标题会有噪声（同一首不同页可能猜成"曲名"或"首句歌词"），
    汇总到组一级才稳定，左侧那份谱的名字也就干净了。
    """
    d = _read_all()
    gs = d.setdefault("groups", {})
    buckets: Dict[str, List[str]] = {}
    for it in d["items"].values():
        g = it.get("group")
        t = (it.get("title") or "").strip()
        if g and t:
            buckets.setdefault(g, []).append(t)
    n = 0
    for g, titles in buckets.items():
        cnt: Dict[str, int] = {}
        for t in titles:
            cnt[t] = cnt.get(t, 0) + 1
        best = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        node = gs.setdefault(g, dict(id=g, created=time.strftime("%Y-%m-%d %H:%M:%S")))
        if node.get("name_locked"):
            continue          # 目录名/手工改过的名字优先，别被 AI 标题盖掉
        cur = node.get("name") or ""
        if best and cur != best:
            node["name"] = best
            n += 1
    d["groups"] = gs
    _write_all(d)
    log("按 AI 标题更新了 %d 个组名" % n)
    return dict(updated=n, groups=len(gs))


def merge_group_names(log=print) -> dict:
    """把「X 1」「X 2」这种编号组并回已有的「X」组；同名的组也合并。

    需求：认不出标题的页按目录名建组时，如果目录名本来就是一个已存在的曲谱，
    应该并进那一份，而不是另开「世末积雨云 1」。
    """
    d = _read_all()
    items = d["items"]
    gs = d.get("groups") or {}
    if not gs:
        return dict(merged=0)
    norm = {gid: _norm_title(g.get("name") or "") for gid, g in gs.items()}

    # 1) 同名合并（规范化后相同）
    keep = {}
    for gid, n in norm.items():
        keep.setdefault(n, gid)
    remap = {gid: keep[norm[gid]] for gid in gs if norm[gid] in keep}

    # 2) "X 1"/"X 2" 并回 "X"
    by_name = {}
    for gid, g in gs.items():
        by_name.setdefault(_norm_title(g.get("name") or ""), gid)
    for gid, g in list(gs.items()):
        m = re.match(r"^(.*?)\s+(\d{1,2})$", (g.get("name") or "").strip())
        if not m:
            continue
        base = _norm_title(m.group(1))
        if base in by_name and by_name[base] != remap.get(gid, gid):
            remap[gid] = by_name[base]

    changed = sum(1 for gid, tgt in remap.items() if tgt != gid)
    if not changed:
        log("没有需要合并的组名")
        return dict(merged=0)

    # 应用映射
    for gid, tgt in remap.items():
        if gid == tgt:
            continue
        tname = gs[tgt].get("name")
        for it in items.values():
            if it.get("group") == gid:
                it["group"] = tgt
                it["group_title"] = tname
                it["group_n"] = 0
    # 重算页数、删空组
    cnt = {}
    for it in items.values():
        if it.get("deleted"):
            continue
        g = it.get("group")
        if g:
            cnt[g] = cnt.get(g, 0) + 1
    for it in items.values():
        g = it.get("group")
        it["group_n"] = cnt.get(g, 0) if g else 0
    for gid in list(gs):
        if gid not in cnt:
            gs.pop(gid, None)
    # 组内顺序按时间重排
    for gid in gs:
        pages = sorted([it for it in items.values() if it.get("group") == gid],
                       key=lambda x: (x.get("t0") or 0, x.get("id") or ""))
        for pos, it in enumerate(pages, 1):
            it["group_pos"] = pos
    d["groups"] = gs
    _write_all(d)
    log("合并了 %d 个组（并回已有的同名/同曲谱组）" % changed)
    return dict(merged=changed)


def group_untitled(ids=None, name_prefix: str = "未命名", log=print) -> dict:
    """把「认不出歌名」的页也建成独立的曲谱分组（放到左侧）。

    需求：右侧那些其实多是新曲子，不能一直堆在未分类里。
    聚类：按目录分组，目录内按时间排序，间隔 > μ+1.5σ 就断开；
    同一目录再并一次（同目录 + 相邻且时间接近）。
    组名：`未命名 N`；如果目录名本身像歌名（不是"保存简谱"这类），就用目录名 + 序号。
    """
    backfill_times()
    d = _read_all()
    items = d["items"]
    seq = [it for it in items.values()
           if not it.get("deleted") and not it.get("group") and not (it.get("title") or "").strip()
           and (ids is None or it["id"] in set(ids))]
    if not seq:
        log("没有需要分组的未识别页")
        return dict(groups=0, pages=0)

    by_dir = {}
    for it in seq:
        dirn = os.path.dirname(it.get("origin") or it.get("image") or "").lower()
        by_dir.setdefault(dirn, []).append(it)

    clusters = []
    for dirn, its in by_dir.items():
        its.sort(key=lambda x: (x.get("t0") or 0, x.get("id") or ""))
        ts = [x.get("t0") or 0 for x in its]
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        if gaps:
            mu = sum(gaps) / len(gaps)
            sd = (sum((g - mu) ** 2 for g in gaps) / len(gaps)) ** 0.5
            thr = max(10.0, mu + 1.5 * sd)
        else:
            thr = 10.0
        cur = [its[0]]
        for prev, it in zip(its, its[1:]):
            gap = (it.get("t0") or 0) - (prev.get("t0") or 0)
            if gap > thr:
                clusters.append((dirn, cur))
                cur = [it]
            else:
                cur.append(it)
        clusters.append((dirn, cur))

    gs = d.get("groups") or {}
    used_names = {g.get("name") for g in gs.values()}
    made = []
    seqno = 1
    for dirn, its in clusters:
        song = (its[0].get("song") or "").strip()
        base = song if (song and song not in ("保存简谱", "上传", "未命名")) else name_prefix
        name = None
        for _try in range(200):
            cand = base if not used_names or base not in used_names else "%s %d" % (base, seqno)
            if cand not in used_names:
                name = cand
                break
            seqno += 1
        name = name or ("%s %d" % (name_prefix, len(gs) + 1))
        used_names.add(name)
        gid = "g_u" + hashlib.sha1((name + str(len(gs)) + time.strftime("%H%M%S")).encode("utf8")).hexdigest()[:8]
        gs[gid] = dict(id=gid, name=name, name_locked=True, order=len(gs),
                       created=time.strftime("%Y-%m-%d %H:%M:%S"))
        for pos, it in enumerate(its, 1):
            node = items[it["id"]]
            node["group"] = gid
            node["group_title"] = name
            node["group_pos"] = pos
            node["group_n"] = len(its)
        made.append(dict(name=name, n=len(its)))
    d["groups"] = gs
    _write_all(d)
    log("未识别页建成 %d 份新曲谱，共 %d 页" % (len(made), sum(m["n"] for m in made)))
    return dict(groups=len(made), pages=sum(m["n"] for m in made), detail=made)


def dedupe_exact(log=print) -> dict:
    """去掉**完全重复**的页：同一张图（字节一致）或同一份 OCR 文本（同一页换个格式）。

    每组只留一张，其余软删除（进垃圾箱、扫描也不会复活）。保留优先级：
    已归类 > 有标题 > 图片更大 > id 更小。
    返回 {removed, groups, kept}。
    """
    import hashlib as _h
    d = _read_all()
    items = d["items"]
    alive = [it for it in items.values() if not it.get("deleted")]
    keys = {}
    for it in alive:
        sig = []
        try:
            with open(it.get("image") or "", "rb") as f:
                sig.append("img:" + _h.sha1(f.read()).hexdigest())
        except Exception:
            pass
        tp = it.get("ocr_txt") or ""
        if tp and os.path.exists(tp):
            try:
                t = open(tp, encoding="utf8").read().strip()
                if len(t) > 80:            # 文本太短不能当判据
                    sig.append("txt:" + _h.sha1(t.encode("utf8")).hexdigest())
            except Exception:
                pass
        for s in sig:
            keys.setdefault(s, []).append(it)

    seen = set()
    removed = 0
    groups = 0
    kept_ids = []
    for s, group in keys.items():
        ids = {g["id"] for g in group}
        if len(ids) < 2:
            continue
        if ids <= seen:
            continue
        seen |= ids
        groups += 1
        group.sort(key=lambda x: (0 if x.get("group") else 1,
                                  0 if x.get("title") else 1,
                                  -os.path.getsize(x.get("image") or "") if os.path.exists(x.get("image") or "") else 0,
                                  x.get("id") or ""))
        keep = group[0]
        kept_ids.append(keep["id"])
        for it in group[1:]:
            if it.get("deleted"):
                continue
            it["deleted"] = True
            it["deleted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            it["deleted_reason"] = "完全重复"
            removed += 1
    # 组内页数重算 + 清掉空组
    cnt = {}
    for it in items.values():
        if it.get("deleted"):
            it["group"] = ""
            it["group_title"] = ""
            it["group_n"] = 0
            continue
        g = it.get("group")
        if g:
            cnt[g] = cnt.get(g, 0) + 1
    for it in items.values():
        g = it.get("group")
        it["group_n"] = cnt.get(g, 0) if g else 0
    for gid in list((d.get("groups") or {})):
        if gid not in cnt:
            (d.get("groups") or {}).pop(gid, None)
    _write_all(d)
    log("完全重复：%d 组，删掉 %d 页（每组留 1 页）" % (groups, removed))
    return dict(removed=removed, groups=groups, kept=kept_ids)


def ensure_sources_from_origins() -> int:
    """把已有条目里出现过的**原文件所在目录**登记成导入源。

    这样"更新（扫新文件）"对历史导入的目录也立即有效，不用重新导入一次。
    """
    items = load()
    have = {s.get("note") for s in sources()}
    have_paths = [s.get("note") for s in sources()
                  if s.get("kind") == "path" and s.get("note")]
    dirs: Dict[str, int] = {}
    for it in items.values():
        o = it.get("origin") or ""
        if not o or not os.path.exists(o):
            continue
        dirs[os.path.dirname(o)] = dirs.get(os.path.dirname(o), 0) + 1
    n = 0
    for d in sorted(dirs):
        if d in have or _is_inside(d, LIB_DIR):
            continue
        if any(_is_inside(d, p) for p in have_paths):   # 已被某个顶层目录覆盖
            continue
        add_source(os.path.basename(d) or d, "path", note=d)
        have_paths.append(d)
        n += 1
    if n:
        collapse_sources()          # 立刻折叠，别让子目录把清单撑爆
    return n


def get(i: str) -> Optional[dict]:
    return load().get(i)


def update(i: str, **kw) -> Optional[dict]:
    items = load()
    it = items.get(i)
    if not it:
        return None
    it.update({k: v for k, v in kw.items() if k in
               ("title", "key", "target", "status", "notes", "uncovered", "ocr_txt", "out",
                "keys", "multi", "dup", "order", "group", "group_title",
                "group_pos", "group_n", "strength", "no_ai", "manual")})
    it["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(items)
    return it


def detect_key(ocr_text: str) -> str:
    """从谱头 OCR 文本里读 `1=X`，读不到返回空串。"""
    pc = RI.KEY_FROM_OCR(ocr_text or "")
    if pc is None:
        return ""
    return KEY_NAMES[int(pc) % 12]


def ingest_files(files, title: str = "", key: str = "", target: str = "",
                 group_id: str = "", log=print) -> dict:
    """把上传的文件收进库（不做 OCR、不调 AI），返回 ids。

    需求：很多时候谱子是自己画的/已经校对好的，不需要模型再看一遍；
    只要把图放进库、填上谱名和调号，剩下的在纠错页里手动点音符就行。
    """
    if not files:
        return dict(ok=False, error="没有文件")
    title = (title or "").strip()
    key = (key or "").strip()
    target = (target or "").strip()
    gid = (group_id or "").strip()
    if not gid and title:
        g = group_new(title)
        gid = g["id"]
    added, dup, bad, ids = 0, 0, 0, []
    for fn, data in files:
        base = os.path.basename(str(fn).replace("\\", "/"))
        if not base.lower().endswith(IMG_EXT):
            bad += 1
            continue
        try:
            i, created = import_uploaded(base, data, song=title or os.path.splitext(base)[0],
                                         key=key)
        except Exception as e:
            log("人工导入失败 %s：%s" % (base, str(e)[:80]))
            bad += 1
            continue
        ids.append((i, base))
        added += 1 if created else 0
        dup += 0 if created else 1
        fields = dict(status="new", no_ai=True, manual=True)
        if title:
            fields["title"] = title
        if key:
            fields["key"] = key
        if target:
            fields["target"] = target
        if gid:
            fields.update(group=gid, group_title=title or "")
        update(i, **fields)
    # 人工导入是"一份谱多页"：按文件名排序定页码（第 1、2、3 页）
    ids.sort(key=lambda t: t[1])
    ids = [t[0] for t in ids]
    if gid:
        for pos, i in enumerate(ids, 1):
            update(i, group=gid, group_title=title or "", group_pos=pos,
                   group_n=len(ids))
        gs = groups()
        n = sum(1 for it in load().values() if it.get("group") == gid and not it.get("deleted"))
        for k, g in gs.items():
            if k == gid:
                g["n"] = n
        _write_all(dict(items=load(), groups=gs,
                        sources=_read_all().get("sources") or [],
                        trash=_read_all().get("trash") or []))
    log("收文件：新增 %d，已存在 %d，失败 %d（组 %s）" % (added, dup, bad, gid or "-"))
    return dict(ok=True, added=added, dup=dup, bad=bad, ids=ids, group=gid,
                title=title)


def import_manual(files, title: str = "", key: str = "", target: str = "",
                  group_id: str = "", log=print) -> dict:
    """人工导入 = **标题/调号由人填，OCR 照常跑**（不调 AI）。

    和"AI 批量导入"的区别只有一点：不联网猜歌名。
    """
    r = ingest_files(files, title=title, key=key, target=target,
                     group_id=group_id, log=log)
    ids = r.get("ids") or []
    if ids:
        log("人工导入：开始 OCR（%d 页）" % len(ids))
        ocr_batch_ids(ids, mode="max", log=log)
        for i in ids:
            upd = {}
            it = get(i) or {}
            if title:
                upd["title"] = title
            if key:
                upd["key"] = key
            if target:
                upd["target"] = target
            upd["no_ai"] = True
            upd["manual"] = True
            if it.get("ocr_txt"):
                upd["status"] = "done" if it.get("out") else "ocr"
            if upd:
                update(i, **upd)
        log("人工导入完成：%d 页（OCR 已跑，未调 AI）" % len(ids))
    r["ocr"] = len(ids)
    return r


def ensure_ocr(it: dict, force: bool = False) -> str:
    """拿到这一页的 OCR 文本（没有就跑本地 Xiaomi OCR 并缓存）。"""
    p = it.get("ocr_txt") or ocr_path_for_id(it["id"])
    if os.path.exists(p) and not force:
        return open(p, encoding="utf8").read()
    if not XO.available():
        raise RuntimeError("Xiaomi-OCR 不可用，无法为新图片生成 OCR 文本")
    raw = ""
    txt = ""
    for attempt in range(3):
        raw = MO.sanitize_ocr(XO.ocr_image(it["image"]))
        trimmed, cut = MO.trim_degenerate(raw)
        txt = trimmed if cut else raw
        if not cut:
            break
    txt = ensure_music_text(it, txt)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf8").write(txt)
    update(it["id"], ocr_txt=p, status="ocr")
    return txt


def ocr_batch_ids(ids, mode: str = "serial", log=print) -> dict:
    """批量 OCR：把缺文本的页一起送本地模型。

    mode = serial / cpu_gpu / gpu2，见 `xiaomi_ocr.ocr_batch`。
    """
    from . import merge_octave as MO
    from . import xiaomi_ocr as XO
    its = [it for it in (get(i) for i in ids) if it]
    todo = [it for it in its if not (it.get("ocr_txt") and os.path.exists(it["ocr_txt"]))]
    if not todo:
        log("都有 OCR 文本了，跳过 %d 页" % len(its))
        return dict(done=0, total=0, skipped=len(its))
    if not XO.available():
        raise RuntimeError("Xiaomi-OCR 不可用")
    log("批量 OCR：%d 页，方式 %s" % (len(todo), mode))
    by_img = {it["image"]: it for it in todo}
    saved = [0]

    def _save(path, raw):
        it = by_img.get(path)
        if not it:
            return
        txt, _cut = MO.trim_degenerate(MO.sanitize_ocr(raw))
        if not txt.strip():
            return
        txt = ensure_music_text(it, txt, log=lambda m: log(m))
        dst = ocr_path_for_id(it["id"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf8").write(txt)
        update(it["id"], ocr_txt=dst, status=("done" if it.get("out") else "ocr"))
        saved[0] += 1

    XO.ocr_batch([it["image"] for it in todo], mode=mode, log=log, on_result=_save)
    ok = saved[0]
    log("批量 OCR 完成 %d/%d" % (ok, len(todo)))
    return dict(done=ok, total=len(todo), skipped=len(its) - len(todo))


def music_line_count(txt: str) -> int:
    """数一下文本里"像谱行"的行数。

    **必须排除调号/速度行**：`1 = D 4/4`、`♩ = 112` 里也有好几个数字，
    之前把它们算成谱行，导致 51 页里像「纯蓝」「越冰船」这些只有调号没有谱行的页
    被误判成"已经有谱行"而漏掉兜底修复。
    """
    import re as _re
    n = 0
    for ln in (txt or "").splitlines():
        s0 = ln.strip()
        if not s0:
            continue
        if _re.search(r"1\s*=", s0) or "♩" in s0 or "=" in s0:
            continue          # 调号 / 速度标记
        if sum(1 for ch in s0 if ch.isdigit()) >= 3:
            n += 1
    return n


def rapid_music_lines(image: str):
    """用 RapidOCR 的字符框，按行拼出数字谱行。

    VLM（Xiaomi-OCR）对带减时线/八度点的密集简谱，经常只吐出歌词、
    把数字行整行漏掉（万神纪 3508px 原图也一样）。RapidOCR 对数字很稳，
    这里用它的框重建"一行一串数字"，再交给 merge_page 对齐。
    """
    from . import ocr_boxes as OB
    import cv2
    bgr = decode_bgr(image)
    if bgr is None:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes, _s = OB.read_boxes(gray)
    chars = OB.chars_from_boxes(boxes, digits_only=False)
    digits = [c for c in chars if str(c.ch).isdigit()]
    if not digits:
        return []
    hs = sorted(c.h for c in digits)
    med = hs[len(hs) // 2] or 10
    rows = []
    for c in sorted(digits, key=lambda c: c.cy):
        for r in rows:
            if abs(r["y"] - c.cy) <= 0.6 * med:
                r["items"].append(c)
                r["y"] = sum(x.cy for x in r["items"]) / len(r["items"])
                break
        else:
            rows.append(dict(y=c.cy, items=[c]))
    rows.sort(key=lambda r: r["y"])
    out = []
    for r in rows:
        r["items"].sort(key=lambda c: c.cx)
        if len(r["items"]) >= 2:
            out.append(" ".join(str(c.ch) for c in r["items"]))
    return out


def ensure_music_text(it: dict, txt: str, log=print) -> str:
    """如果这段 OCR 文本几乎没有谱行，就用 RapidOCR 补一份数字行。

    返回最终要写入 ocr.txt 的文本（数字行在前，原文歌词在后）。
    """
    n = music_line_count(txt)
    if n >= 2:
        return txt
    try:
        lines = rapid_music_lines(it.get("image") or "")
    except Exception as e:
        log("RapidOCR 兜底失败：%s" % str(e)[:100])
        return txt
    if len(lines) < 2:
        return txt
    log("VLM 只出 %d 行谱行 → RapidOCR 兜底补了 %d 行" % (n, len(lines)))
    return "\n".join(lines) + "\n\n" + (txt or "")


def do_ocr(i: str) -> dict:
    """只跑 OCR，把文本缓存下来（本地模型，不花 token，也不出图）。"""
    it = get(i)
    if not it:
        raise RuntimeError("乐谱不存在: %s" % i)
    ensure_ocr(it)
    # 已经出过图的页不许被"OCR"降级回 ocr
    st = "done" if (it.get("out") and os.path.exists(it["out"])) else "ocr"
    return update(i, status=st) or it


def _segments_for(page, pairs, keys: List[dict], txt: str) -> List[int]:
    """把「第 N 个 `1=X`」折算成逐音调号（中途转调）。

    OCR 文本没有坐标，所以按 `行号 / 总行数 × 页高` 估一个分界高度，
    再按"音的中心 y 在该分界之后就用新调"切段——和纠错页里点锚定的规则一致。
    """
    H = float(page.gray.shape[0])
    nlines = max(1, len((txt or "").splitlines()))
    pts = []
    for k in keys[1:]:
        pc = M.parse_key(k["key"])
        if pc is not None:
            pts.append((H * float(k["line"]) / nlines, int(pc)))
    if not pts:
        return []
    first = M.parse_key(keys[0]["key"])
    out = []
    for s, _o in pairs:
        cy = (s.box[1] + s.box[3]) / 2.0
        cur = 0 if first is None else int(first)
        for yf, pc in pts:
            if cy >= yf:
                cur = pc
        out.append(cur)
    return out


def process(i: str, log=print) -> dict:
    """跑完整流程：OCR → 合并 → 渲染红字图。返回更新后的记录。"""
    from .glyphnet import GlyphNet
    it = get(i)
    if not it:
        raise RuntimeError("乐谱不存在: %s" % i)
    t0 = time.time()
    txt = ensure_ocr(it)
    it = get(i) or it
    keys = it.get("keys") or detect_keys(txt)
    key_name = it.get("key") or (keys[0]["key"] if keys else "") or detect_key(txt) or "C"
    key_pc = M.parse_key(key_name)
    if key_pc is None:
        key_pc = 0
    target_name = it.get("target") or KEY_NAMES[key_pc]
    ref_pc = M.parse_key(target_name)
    if ref_pc is None:
        ref_pc = key_pc
    net = GlyphNet()
    strength = int(it.get("strength") if it.get("strength") is not None else 1)
    lines, pairs, stat, page = MO.merge_page(it["image"], txt, net, strength=strength)
    notes = [n for l in lines for n in l]
    pitches = [M.note_to_abs(int(n.digit), n.acc, int(n.octave), key_pc)
               for n in notes if n.digit != "0"]
    if not it.get("target") and pitches:
        # 没指定目标调时，沿用旧行为：自动选升降号少的记谱
        ref_pc, _reason = M.pick_reference(pitches, 3)
        target_name = KEY_NAMES[int(ref_pc) % 12]
    segs = None
    if it.get("multi") and len(keys) > 1:
        segs = _segments_for(page, pairs, keys, txt)
    bgr, st = EF.render_fixed_image(page, pairs, key_pc, ref_pc, keys=segs or None)
    dst = it.get("out") or out_path_for(it["song"], it["image"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    core.imwrite(dst, bgr)
    dt = time.time() - t0
    log("%s → 1=%s%s 目标1=%s 音%d 休止%d 漏%d 一致%.0f%% %.0fs"
        % (it.get("song"),
           "→".join([keys[0]["key"]] + [k["key"] for k in keys[1:]]) if keys else key_name,
           "(中途转调)" if segs else "",
           target_name, st.get("total", 0), st.get("rest", 0),
           st.get("uncovered", 0), stat.get("agree_rate", 0) * 100, dt))
    return update(i, key=key_name, target=target_name, status="done", out=dst,
                  keys=keys, notes=int(st.get("total", 0)),
                  uncovered=int(st.get("uncovered", 0))) or it


def stats() -> dict:
    items = load()
    by = {"new": 0, "ocr": 0, "done": 0, "error": 0}
    for it in items.values():
        by[it.get("status", "new")] = by.get(it.get("status", "new"), 0) + 1
    return dict(total=len(items), **by)
