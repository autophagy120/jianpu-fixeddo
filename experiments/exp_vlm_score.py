"""VLM 结构化识谱 —— 第二版：告诉它图像尺寸、强制直接输出 JSON、给足 token。"""
import base64
import json
import os
import ssl
import sys
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, ".")
import cv2

from jp import core

KEY = os.environ.get("DS_KEY", "")
BASE = "https://api.deepseek.com"


def ask(img_bgr, max_tokens=8000, model="deepseek-flash"):
    h, w = img_bgr.shape[:2]
    prompt = (
        f"这是一条简谱谱行的图片，宽 {w} 像素、高 {h} 像素（像素坐标已给出，不要推算）。\n"
        "逐音符输出：digit(0-7)、oct(1=上方一点,-1=下方一点,2/-2=两点,0=无点)、"
        "acc(\"\"或\"#\"或\"b\")、x(数字左边缘横坐标整数)、w(数字宽度整数)。\n"
        "只输出真实存在的音符，看不清就不要输出。\n"
        "要求：**第一个字符必须是 {**，只输出一个 JSON 对象，不要任何解释、不要思考过程、不要代码块。\n"
        '格式：{"notes":[{"digit":"6","oct":0,"acc":"","x":31,"w":12}],"line_width":%d}' % w
    )
    ok, enc = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(enc.tobytes()).decode()
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read().decode())
    msg = body["choices"][0]["message"]
    return (msg.get("content") or "").strip(), (msg.get("reasoning_content") or "").strip(), body.get("usage", {})


def parse(txt):
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        return None
    for cut in (j + 1,):
        try:
            return json.loads(txt[i:cut])
        except Exception:
            pass
    return None


def run(tag, path, y0, y1, x0, x1, scale):
    bgr = core.load_bgr(path)
    crop = bgr[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    content, reason, usage = ask(big)
    data = parse(content) or parse(reason)
    print(f"\n===== {tag}  图 {big.shape[1]}x{big.shape[0]}  usage={usage.get('completion_tokens')} =====")
    if not data:
        print("  解析失败。content 前 200:", content[:200].replace("\n", " "))
        print("  reasoning 前 200:", reason[:200].replace("\n", " "))
        return
    notes = data.get("notes", [])
    seq = []
    for n in notes:
        a = n.get("acc", "") or ""
        o = int(n.get("oct", 0) or 0)
        seq.append(f"{a}{n.get('digit', '?')}" + ("^" * o if o > 0 else "v" * (-o) if o < 0 else ""))
    print(f"  {len(notes)} 个：", " ".join(seq))
    print("  前3个原始项:", json.dumps(notes[:3], ensure_ascii=False))


if __name__ == "__main__":
    p3 = r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"
    # 第 1 行左半段：真值 ( 6 1 2 3  6 1 2 3  6 1 2 3
    run("三月雨 第1行左半 (真值 6123 x3)", p3, 126, 174, 20, 300, 5.0)
