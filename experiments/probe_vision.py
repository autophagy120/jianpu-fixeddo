"""Probe whether the DeepSeek endpoint accepts image input (vision or not)."""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np

KEY = os.environ.get("DS_KEY", "")
BASE = "https://api.deepseek.com"

img_path = sys.argv[1] if len(sys.argv) > 1 else "out/qa_crop2.png"
buf = np.fromfile(img_path, dtype=np.uint8)
im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
if im.shape[1] > 700:
    s = 700 / im.shape[1]
    im = cv2.resize(im, (700, int(im.shape[0] * s)))
ok, enc = cv2.imencode(".png", im)
b64 = base64.b64encode(enc.tobytes()).decode()

for model in ("deepseek-flash", "deepseek-v4-pro"):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "这条简谱行里有几个数字音符？只回一个数字。"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            ],
        }],
        "max_tokens": 40,
        "temperature": 0,
    }
    req = urllib.request.Request(BASE + "/chat/completions",
                                data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        print(model, "-> VISION OK:", body["choices"][0]["message"]["content"][:120])
    except urllib.error.HTTPError as e:
        print(model, "-> HTTP", e.code, e.read().decode()[:200])
    except Exception as e:
        print(model, "-> fail", type(e).__name__, str(e)[:120])
