"""Xiaomi-OCR-0 本地调用 + 简谱文本解析（零 API 成本）。

模型：SeerRay-Lab/Xiaomi-OCR-0（0.8B 文档解析），GGUF Q4_K_M + mmproj，llama.cpp 本地跑。
实测：三月雨整页（742×1003）在 GTX 1650 上约 1 分钟出全文，数字与歌词都正确。

输出约定（实测得到的写法）：
    6i23      -> 6  1̇  2  3      （'i' 是"带高音点的 1"）
    2·3       -> 2  3            （· 是附点，不影响音高）
    -         -> 增时线，跳过
    |  ( )  :::  ||  -> 结构符号
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

OCR_HOME = os.environ.get("XIAOMI_OCR_HOME", r"E:\jianpu-env\ocr")
CLI = os.path.join(OCR_HOME, os.environ.get("XIAOMI_OCR_BIN", "llama-cuda"), "llama-mtmd-cli.exe")
# CPU 专用二进制（没有就退回 GPU 二进制 + -ngl 0）
CLI_CPU = os.path.join(OCR_HOME, "llama", "llama-mtmd-cli.exe")
if not os.path.exists(CLI_CPU):
    CLI_CPU = CLI
MODEL = os.path.join(OCR_HOME, "model", "Xiaomi-OCR-0.Q4_K_M.gguf")
MMPROJ = os.path.join(OCR_HOME, "model", "mmproj-bf16.gguf")
# 提示词必须点名要简谱记号：用英文通用提示时，模型只转写歌词、把整行数字漏掉
# （实测同一张图：通用 prompt 得 2 行数字行，下面这条得 12 行）
PROMPT = ("请把图片里的简谱内容完整转成文字：数字音符（1-7）、八度点、减时线、歌词、调号都要，按行输出。")
PROMPT_GENERIC = "Extract the text in the image."


def available() -> bool:
    return os.path.exists(CLI) and os.path.exists(MODEL) and os.path.exists(MMPROJ)


def ocr_image(path: str, max_tokens: int = 2400, gpu_layers: int = 99, timeout: int = 900,
              target_width: int = 1800, threads: int = 0, prompt: str = "") -> str:
    """对一张图跑 Xiaomi-OCR-0，返回纯文本。

    `target_width`：先把图缩放到这个宽度再喂模型。低分辨率扫描（三月雨只有
    742 px 宽、数字 9 px）在小图上会**退化**——实测同一张图直接喂，模型会连着
    吐 100 多行 `啊！`、把后半页谱行全吞掉（音符数 423 → 187），重试三次都一样；
    放大到 1800 px 宽后输出稳定。本来就够大的图不会被放大。
    """
    if not available():
        raise RuntimeError(f"Xiaomi-OCR-0 未就绪：{CLI} / {MODEL} / {MMPROJ}")
    src = path
    tmp = None
    try:
        if target_width:
            img = _load_bgr(path)
            h, w = img.shape[:2]
            if w < target_width * 0.95:
                k = target_width / float(w)
                img = cv2.resize(img, (int(round(w * k)), int(round(h * k))),
                                 interpolation=cv2.INTER_LANCZOS4)
                # 加唯一后缀：并行跑时两张同尺寸的图会撞到同一个临时文件，
                # 结果就是 A 页拿到 B 页的 OCR 文本（之前 OCR 缓存撞名就是这么坑的）
                import threading
                tmp = os.path.join(tempfile.gettempdir(),
                                   "jp_ocr_%dx%d_%d_%d.png" % (img.shape[1], img.shape[0],
                                                               os.getpid(), threading.get_ident()))
                ok, buf = cv2.imencode(".png", img)
                if ok:
                    buf.tofile(tmp)
                    src = tmp
    except Exception:
        src = path
    exe = CLI if gpu_layers > 0 else CLI_CPU
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(exe) + os.pathsep + env.get("PATH", "")
    cmd = [exe, "-m", MODEL, "--mmproj", MMPROJ, "--image", src,
           "-p", (prompt or PROMPT), "-n", str(max_tokens), "--temp", "0",
           "-ngl", str(gpu_layers)]
    if threads > 0:
        cmd += ["-t", str(int(threads))]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf8", errors="replace",
                       env=env, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    # 去掉 <think>...</think> 与 llama.cpp 的日志行
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    lines = []
    for ln in out.splitlines():
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if re.match(r"^[0-9.]+ [IWE] ", s):       # llama.cpp 日志
            continue
        if s.startswith(("llama_", "main:", "encode_", "mtmd", "ggml_", "load_", "clip_")):
            continue
        lines.append(s)
    txt = "\n".join(lines).strip()
    return txt


def _is_oom(err: str) -> bool:
    e = (err or "").lower()
    return ("out of memory" in e) or ("cuda error" in e) or ("failed to allocate" in e)


def _is_oom(err: str) -> bool:
    e = (err or "").lower()
    return ("out of memory" in e) or ("cuda error" in e) or ("failed to allocate" in e)


def plan_workers(mode: str):
    """返回 (gpu_工人数, cpu_工人数, 每个 cpu 工人的线程数)。

    实测（GTX1650 4G）：
    - GPU 单张 ~40-170s，是主力；
    - **一个 CPU 进程就会把 12 线程吃满**（llama.cpp 默认 -t=核数），单张 400s+；
    - 3 个 GPU 进程会把 4G 显存挤爆（3.7G/4G），单张反而 600-750s。
    所以默认 `gpu_first`：**1 GPU + 1 CPU**，CPU 只做"填料"，做不完的由 GPU 接管。
    """
    import os as _os
    g = int(_os.environ.get("OCR_GPU_WORKERS", "0") or 0)
    c = int(_os.environ.get("OCR_CPU_WORKERS", "0") or 0)
    t = int(_os.environ.get("OCR_CPU_THREADS", "0") or 0)
    if mode == "serial":
        return 1, 0, 0
    if mode == "gpu2":
        return (g or 2), 0, 0
    if mode == "cpu_gpu":
        return (g or 1), (c or 1), (t or 6)
    if mode == "max":
        return (g or 1), (c or 3), (t or 3)
    # gpu_first（默认）：1 GPU + 1 CPU，CPU 线程压到 4，别把整机吃满
    return (g or 1), (c or 1), (t or 4)


STEAL_AFTER = 0.0       # GPU 一空闲就复制 CPU 正在跑的那张（双跑，先完成者胜）


def ocr_batch(paths, mode: str = "gpu_first", max_tokens: int = 2400,
              target_width: int = 1800, log=print, on_result=None) -> dict:
    """批量 OCR。

    mode：
      - **gpu_first（默认）**：1 GPU + 1 CPU。GPU 先抢队列；CPU 只拿一件慢慢啃；
        队列空了而 CPU 那张还没完（> STEAL_AFTER 秒）时，**GPU 接管这张**（重跑），
        CPU 的结果作废 —— 也就是"CPU 跑不完的自动调度给 GPU"。
      - serial / cpu_gpu / gpu2 / max：旧模式的固定工人数。
    """
    import queue
    import threading
    import time as _t

    results = {}
    if not paths:
        return results
    ng, nc, ct = plan_workers(mode)
    gpu_first = (mode == "gpu_first")
    log("OCR 调度：%d GPU + %d CPU（CPU %d 线程）%s，任务 %d 页"
        % (ng, nc, ct, "，GPU 优先/可接管" if gpu_first else "", len(paths)))

    q = queue.Queue()
    for p in paths:
        q.put(p)
    lock = threading.Lock()
    inflight = {}          # path -> 开始时间（只有 CPU 在跑）
    stolen = set()
    done = [0]
    t0 = _t.time()
    finished = threading.Event()

    def save(p, txt, err, tag):
        with lock:
            if p in results:          # 双跑：另一边已经先完成
                log("[%s] %s 另一路已先完成，丢弃这份" % (tag, p[-28:]))
                return
            if p in stolen and tag.startswith("CPU"):
                log("[%s] %s 已被 GPU 抢跑，结果丢弃" % (tag, p[-28:]))
                return
            results[p] = txt
            done[0] += 1
            if on_result is not None:
                try:
                    on_result(p, txt)
                except Exception as e:
                    log("落盘失败：%s" % str(e)[:100])
            el = _t.time() - t0
            log("[%s] (%d/%d) %.0fs%s" % (tag, done[0], len(paths), el,
                                         ("  失败：" + err) if err else ""))
            if done[0] >= len(paths):
                finished.set()

    def run_one(p, tag, gpu_layers, threads):
        try:
            txt = ocr_image(p, max_tokens=max_tokens, gpu_layers=gpu_layers,
                            target_width=target_width, threads=threads)
            err = ""
        except Exception as e:
            txt, err = "", "%s: %s" % (type(e).__name__, str(e)[:140])
        if not txt and gpu_layers > 0 and _is_oom(err):
            try:
                txt = ocr_image(p, max_tokens=max_tokens, gpu_layers=0,
                                target_width=target_width, threads=ct or 4)
                err = ""
            except Exception as e:
                err = str(e)[:140]
        save(p, txt, err, tag)

    def gpu_worker(idx):
        tag = "GPU%d" % (idx + 1)
        while not finished.is_set():
            try:
                p = q.get_nowait()
                run_one(p, tag, 99, 0)
                continue
            except queue.Empty:
                pass
            if not gpu_first:
                return
            # 队列空了 → GPU 反正闲着：直接把 CPU 正在跑的那张复制一份双跑，
            # 谁先出结果谁算（CPU 慢，基本都会被 GPU 抢先）。
            cand = None
            with lock:
                now = _t.time()
                for p, st in inflight.items():
                    if p not in stolen and now - st >= STEAL_AFTER:
                        cand = p
                        break
                if cand:
                    stolen.add(cand)
            if cand:
                log("[%s] GPU 空闲 → 和 CPU 并行跑同一页（先完成者胜）" % tag)
                run_one(cand, tag, 99, 0)
                continue
            with lock:
                if done[0] >= len(paths):
                    return
                busy = bool(inflight)
            if busy:
                _t.sleep(0.4)
                continue
            return

    def cpu_worker(idx):
        tag = "CPU%d" % (idx + 1)
        while not finished.is_set():
            try:
                p = q.get_nowait()
            except queue.Empty:
                return
            with lock:
                inflight[p] = _t.time()
            try:
                run_one(p, tag, 0, ct)
            finally:
                with lock:
                    inflight.pop(p, None)

    ths = [threading.Thread(target=gpu_worker, args=(i,), daemon=True) for i in range(ng)]
    ths += [threading.Thread(target=cpu_worker, args=(i,), daemon=True) for i in range(nc)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    log("批量 OCR 结束：%d 页，用时 %.0fs（%.1fs/页）"
        % (len(paths), _t.time() - t0, (_t.time() - t0) / max(1, len(paths))))
    return results


def _load_bgr(path: str):
    """`cv2.imread` 不支持非 ASCII 路径，走 imdecode。"""
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 解析
@dataclass
class OcrNote:
    digit: str
    octave: int = 0
    acc: str = ""
    aug_dot: bool = False
    x_slot: int = 0          # 在该行内的序号（用于落位）
    lyric: str = ""


@dataclass
class OcrLine:
    raw: str
    notes: List[OcrNote] = field(default_factory=list)
    lyric: str = ""
    is_music: bool = False


MUSIC_CHARS = set("01234567i·-|()[]:∪")


def parse_line(text: str) -> OcrLine:
    """把一行 OCR 文本解析成音符序列；含数字比例高的行判为谱行。"""
    line = OcrLine(raw=text)
    buf: List[OcrNote] = []
    pending_acc = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "#♯":
            pending_acc = "#"
            i += 1
            continue
        if ch in "b♭":
            # 'b' 可能是降号，也可能是歌词；只在后面跟数字时当降号
            if i + 1 < n and text[i + 1] in "01234567i":
                pending_acc = "b"
                i += 1
                continue
        if ch in "01234567":
            note = OcrNote(digit=ch, acc=pending_acc)
            pending_acc = ""
            # 紧跟的 i 表示"高八度的 1"（实测写法：6i = 6 1̇）
            if i + 1 < n and text[i + 1] == "i":
                buf.append(note)
                buf.append(OcrNote(digit="1", octave=1))
                i += 2
                continue
            buf.append(note)
            i += 1
            continue
        if ch == "i":
            buf.append(OcrNote(digit="1", octave=1))
            pending_acc = ""
            i += 1
            continue
        if ch == "·":
            if buf:
                buf[-1].aug_dot = True
            i += 1
            continue
        if ch == "-":
            i += 1
            continue
        i += 1
    for k, nt in enumerate(buf):
        nt.x_slot = k
    line.notes = buf
    digits = sum(1 for c in text if c in "01234567i")
    # 谱行判定。原来要求 `>=4 个音`，把**音符很少的小字行**（和声/装饰音，
    # 如 `(3) 3 · 4 4 1 |`，只有 3~5 个音）整行滤掉了——几何侧能看见这些行，
    # 但 OCR 侧不给数字，于是它们全落到"橙框"里、一个红字都不出。
    # 现在放宽到 >=2 个音，但要求行内出现简谱符号（小节线/减时线/附点/括号），
    # 避免把含两个阿拉伯数字的歌词行误判成谱行。
    has_music_sym = any(c in "|｜‖-—·()（）∪" for c in text)
    line.is_music = digits >= 2 and (len(buf) >= 4 or has_music_sym)
    return line


def parse_document(text: str) -> Tuple[List[OcrLine], List[OcrLine]]:
    """拆成 (谱行, 歌词行)。歌词行 = 紧跟在谱行后面的非谱行。"""
    lines = [parse_line(t) for t in text.splitlines()]
    music = [l for l in lines if l.is_music]
    return music, [l for l in lines if not l.is_music]


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else r"E:\jianpu-env\ocr\test\page.png"
    txt = ocr_image(img)
    music, other = parse_document(txt)
    print(f"谱行 {len(music)} 行，其它 {len(other)} 行")
    for k, l in enumerate(music[:3]):
        seq = " ".join(f"{n.acc}{n.digit}{'^'*n.octave}{'·' if n.aug_dot else ''}" for n in l.notes)
        print(f"  行{k} ({len(l.notes)} 音): {seq[:150]}")
