"""VLM 逐片读谱行（按已验证成功的参数：~280px 宽 × 5 倍放大/片）。"""
from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, ".")
import cv2
import numpy as np

from jp import core

KEY = os.environ.get("DS_KEY", "")
BASE = "https://api.deepseek.com"

PROMPT = (
    "这是一条简谱谱行的**一个片段**，图片宽 __W__ 像素、高 __H__ 像素（尺寸已给出，不要推算）。\n"
    "逐音符输出：digit(0-7)、oct(1=上方一点,-1=下方一点,2/-2=两点,0=无点)、"
    "acc(\"\"或\"#\"或\"b\")、x(数字左边缘的横坐标整数，相对这张图)、w(数字宽度整数)。\n"
    "只输出真实存在的音符。不要逐条分析，直接给答案。\n"
    "只输出一个 JSON，第一个字符必须是左花括号：\n"
    '{"notes":[{"digit":"6","oct":0,"acc":"","x":31,"w":12}]}'
)


def ask(img, model="deepseek-flash", max_tokens=9000, timeout=240, dbg=False):
    h, w = img.shape[:2]
    ok, enc = cv2.imencode(".png", img)
    b64 = base64.b64encode(enc.tobytes()).decode()
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.replace("__W__", str(w)).replace("__H__", str(h))},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    msg = body["choices"][0]["message"]
    txt = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        if dbg:
            print(f"      [无JSON] 长度{len(txt)} 结尾: ...{txt[-160:]!r}", flush=True)
        return []
    try:
        data = json.loads(txt[i:j + 1])
    except Exception as exc:
        if dbg:
            print(f"      [JSON坏] {exc} 片段: {txt[i:i+160]!r}", flush=True)
        return []
    return data.get("notes", [])


def read_line(bgr, y0, y1, tile_px=280, overlap=45, scale=5.0, retries=2, dbg=True):
    H, W = bgr.shape[:2]
    notes = []
    x = 0
    ti = 0
    while x < W:
        ti += 1
        x2 = min(W, x + tile_px)
        crop = bgr[max(0, y0):min(H, y1), x:x2]
        if crop.shape[1] < 60:
            break
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        got = []
        for t in range(retries + 1):
            got = ask(big, dbg=(t == retries))
            if got:
                break
        if dbg:
            print(f"    片{ti} x={x}..{x2} -> {len(got)} 个: "
                  f"{' '.join(n.get('digit','?') for n in got[:14])}", flush=True)
        for n in got:
            try:
                nx = x + float(n.get("x", -1)) / scale
                notes.append({"digit": str(n.get("digit")), "oct": int(n.get("oct", 0) or 0),
                              "acc": (n.get("acc") or ""), "x": nx,
                              "w": float(n.get("w", 0)) / scale})
            except Exception:
                continue
        if x2 >= W:
            break
        x = x2 - overlap
    notes.sort(key=lambda d: d["x"])
    # 去重叠区重复
    out = []
    for d in notes:
        if out and abs(d["x"] - out[-1]["x"]) < max(2.0, 0.4 * d.get("w", 3)):
            continue
        out.append(d)
    return out


def main():
    P = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
    bgr = core.load_bgr(P)
    print("=== 三月雨 第 1 行（真值：6123 循环）===")
    notes = read_line(bgr, 126, 172)
    seq = [n["digit"] + ("^" * n["oct"] if n["oct"] > 0 else "v" * (-n["oct"])) for n in notes]
    print(f"\n共 {len(notes)} 个音符：")
    print(" ", " ".join(seq))
    gt = ["6", "1", "2", "3"] * 12
    ok = sum(1 for a, b in zip([n["digit"] for n in notes], gt[:len(notes)]) if a == b)
    print(f"\n逐字正确 {ok}/{len(notes)}")


if __name__ == "__main__":
    main()
