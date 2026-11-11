"""DeepSeek API helpers (the endpoint accepts images, but the model is a reasoner:
the answer may arrive in `reasoning_content`, and a small max_tokens truncates it).
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import re
import urllib.request
from typing import Dict, List, Optional

import cv2
import numpy as np

DEFAULT_BASE = "https://api.deepseek.com"

# ---------------------------------------------------------------- 花钱闸门
# 默认禁止调用 VLM。要调用必须显式设置环境变量 JP_ALLOW_VLM=1，
# 并可用 JP_VLM_MAX_CALLS 限制本次进程内最多调用几次（默认 20）。
_BUDGET_GUARD = {"enabled": os.environ.get("JP_ALLOW_VLM") == "1",
                 "calls": 0,
                 "limit": int(os.environ.get("JP_VLM_MAX_CALLS", "20"))}


def budget_check(what: str = "chat") -> None:
    """在每次真正发请求前调用；超限或未授权就抛错，绝不静默烧钱。"""
    if not _BUDGET_GUARD["enabled"]:
        raise RuntimeError(
            "VLM 调用默认关闭（成本保护）。确认要花钱请设置 JP_ALLOW_VLM=1，"
            "并用 JP_VLM_MAX_CALLS 限制次数。")
    if _BUDGET_GUARD["calls"] >= _BUDGET_GUARD["limit"]:
        raise RuntimeError(
            f"VLM 调用已达上限 {_BUDGET_GUARD['limit']} 次（JP_VLM_MAX_CALLS）。"
            "确认后调大该变量再重试。")
    _BUDGET_GUARD["calls"] += 1


def calls_used() -> int:
    return _BUDGET_GUARD["calls"]


def _b64_png(bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", bgr)
    return base64.b64encode(enc.tobytes()).decode()


def _key() -> str:
    k = os.environ.get("DS_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not k:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".deepseek_key")
        if os.path.exists(p):
            k = open(p, encoding="utf8").read().strip()
    if not k:
        raise RuntimeError("未配置 DEEPSEEK key：设置环境变量 DS_KEY 或写到 .deepseek_key")
    return k


def chat(
    prompt: str,
    image_bgr: Optional[np.ndarray] = None,
    model: str = "deepseek-flash",
    base_url: str = DEFAULT_BASE,
    max_tokens: int = 16000,
    temperature: float = 0.0,
    retries: int = 2,
) -> str:
    budget_check("chat")
    content = [{"type": "text", "text": prompt}]
    if image_bgr is not None:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64_png(image_bgr)}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + _key()},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:300]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        msg = body["choices"][0]["message"]
        txt = (msg.get("content") or "").strip()
        if not txt:
            txt = (msg.get("reasoning_content") or "").strip()
            txt = _strip_reasoning(txt)
        if txt:
            return txt
        last = "空回复（max_tokens 可能被思考过程吃掉了）"
        time.sleep(1.0)
    raise RuntimeError(last)


def _strip_reasoning(txt: str) -> str:
    """Pull the answer out of a chain-of-thought block when `content` is empty."""
    for marker in ("最终答案：", "最终答案:", "答案：", "答案:"):
        if marker in txt:
            return txt.split(marker, 1)[1].strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def ocr_lyric_line(bgr: np.ndarray, model: str = "deepseek-flash") -> str:
    prompt = (
        "这是一行简谱下面的歌词。请逐字输出这行歌词（汉字与标点都要），"
        "不要解释、不要加序号、不要翻译，直接给一行纯文本。"
        "如果某处看不清用□代替。"
    )
    txt = chat(prompt, bgr, model=model, max_tokens=2000)
    return txt.splitlines()[0].strip() if txt else ""


NOTE_STRING_PROMPT = (
    "这是一张简谱（数字谱）谱行片段的图片。请把片段里**从左到右**的每个音符写成一串，"
    "音符之间用一个空格分开。\n"
    "符号约定：数字取 0-7；若数字上方有一点点缀，紧跟一个 ^（两个点就写 ^^）；"
    "若数字下方有点，紧跟一个 _；升号写在数字前面用 #，降号写在数字前面用 b。\n"
    "要求：只输出这一串音符本身，不要输出任何其他文字、不要编号、不要标点、不要换行。\n"
    "如果片段里确实没有音符，就什么都不输出。"
)


def read_note_string(img, temperature: float = 0.4, max_tokens: int = 16000,
                     model: str = "deepseek-flash", base_url: str = DEFAULT_BASE):
    """只要音符串（不要坐标）——模型在这上面最稳。返回 token 列表。"""
    import base64 as _b64
    budget_check("note_string")
    ok, enc = cv2.imencode(".png", img)
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": NOTE_STRING_PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                            _b64.b64encode(enc.tobytes()).decode()}}]}],
        "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + _key()})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read().decode())
    msg = body["choices"][0]["message"]
    txt = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    # 宽松解析：从整段文字里抓"最长的一串音符"，不要求整行干净
    TOKEN = re.compile(r"[#b]?[0-7][\^_]{0,2}")
    groups: List[List[str]] = []
    cur: List[str] = []
    for raw in re.split(r"\s+", txt.replace(",", " ").replace("，", " ").replace("|", " | ")):
        for piece in raw.split("|"):
            piece = piece.strip("()[]{}<>\"'`*:;。、")
            if not piece:
                continue
            if TOKEN.fullmatch(piece):
                cur.append(piece)
            else:
                if len(cur) >= 2:
                    groups.append(cur)
                cur = []
    if len(cur) >= 2:
        groups.append(cur)
    best = max(groups, key=len) if groups else []
    if best == ["1^", "2", "3#", "5_", "b7", "0"]:      # 旧示例被抄回来的情况
        best = []
    return best
