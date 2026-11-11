"""试不同 prompt：看数字谱行是被"模型能力"漏掉，还是"prompt 没让它输出"。"""
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
import jp.xiaomi_ocr as XO

PROMPTS = [
    "Extract the text in the image.",
    "请把图片里的简谱内容完整转成文字：数字音符（1-7）、八度点、减时线、歌词、调号都要，按行输出。",
    "Transcribe the numbered musical notation and lyrics. Include all digits (1-7) with their dots.",
]

it = [x for x in L.load().values() if x["id"].startswith("37d477b5")][0]
img = it["image"]

# 先用 XO 的缩放逻辑准备图
bgr = XO._load_bgr(img)
h, w = bgr.shape[:2]
if w < 1800 * 0.95:
    k = 1800 / float(w)
    bgr = cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_LANCZOS4)
tmp = os.path.join(tempfile.gettempdir(), "prompt_test.png")
cv2.imencode(".png", bgr)[1].tofile(tmp)
print("图:", w, "x", h, "→", bgr.shape[1], "x", bgr.shape[0], flush=True)

env = dict(os.environ)
env["PATH"] = os.path.dirname(XO.CLI) + os.pathsep + env.get("PATH", "")
for k, p in enumerate(PROMPTS, 1):
    cmd = [XO.CLI, "-m", XO.MODEL, "--mmproj", XO.MMPROJ, "--image", tmp,
           "-p", p, "-n", "2400", "--temp", "0", "-ngl", "99"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf8",
                       errors="replace", env=env, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    lines = [ln.strip() for ln in out.splitlines()
             if ln.strip() and not re.match(r"^[0-9.]+ [IWE] ", ln.strip())
             and not ln.strip().startswith(("llama_", "main:", "encode_", "mtmd", "ggml_" ,"load_", "clip_"))]
    txt = "\n".join(lines)
    dl = [ln for ln in lines if sum(c.isdigit() for c in ln) >= 3]
    print("\n=== prompt %d：%d 字，%d 行数字行 ===" % (k, len(txt), len(dl)), flush=True)
    print("  prompt:", p[:50], flush=True)
    for ln in dl[:4]:
        print("   数字行:", ln[:70], flush=True)
    if not dl:
        print("   前 200 字:", repr(txt[:200]), flush=True)
