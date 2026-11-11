"""LLM/VLM verification layer (the part the 2018 program could not have).

Design stance: the recogniser is allowed to be *unsure*.  Anything the local
model cannot settle is packed into a small, self-contained request - one text
row crop plus the draft token list - and handed to a vision LLM whose only job
is to return **minimal edits** in a fixed JSON shape.  That keeps the model
inside a narrow, checkable task instead of asking it to "read the whole page".

Works with any OpenAI-compatible endpoint (Qwen-VL / GLM / GPT / Gemini proxy):

    set LLM_BASE_URL=https://api.deepseek.com/v1
    set LLM_API_KEY=sk-...
    set LLM_MODEL=qwen3-vl-30b-a3b-instruct

Without a key the module runs in dry-run mode: it writes the exact request
payload (image included) so the prompt can be inspected and diffed.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "llm_requests")

# Price table, USD per 1M tokens (input, output).  Update from the provider page
# before you quote a number to anyone.
PRICES = {
    "qwen3-vl-30b-a3b": (0.15, 0.60),
    "qwen3-vl-235b": (0.21, 1.90),
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-5": (1.25, 10.00),
    "claude-sonnet-4.5": (3.00, 15.00),
}

SYSTEM_PROMPT = """你是一名简谱（数字简谱）校对员。用户会给你：
1) 一条已切好的谱行图像（可能含上下八度点、减时线、连音线、小节线）；
2) 该行的 OCR 草稿：按从左到右顺序的音符数组，每个元素含 digit/oct/acc/conf。

简谱规则：
- digit ∈ {"0","1","2","3","4","5","6","7"}，0 是休止符；
- acc 为 "" / "#" / "b"，写在数字左侧；
- oct = 1 表示数字上方的点（高八度），-1 表示下方的点（低八度），0 表示无点；
- 数字下方的横线（减时线）不改变音高，不要据此改 digit；
- 歌词汉字永远不是音符，若某框是汉字请放进 reject。

只输出 JSON，不要解释，不要 markdown 代码块。形如：
{"row_id":"r12","notes":[{"i":0,"digit":"6","oct":0,"acc":"","conf":0.9}],
 "missed":[{"after_i":3,"digit":"5","oct":-1,"acc":""}],
 "reject":[7,11],"verdict":"ok|fixed|uncertain","reason":"<=20字"}
其中 notes 只需列出你认为需要修改的项；未列出的项视为保持草稿原样。"""

RESPONSE_HINT = {
    "row_id": "string",
    "notes": [{"i": "int", "digit": "0-7", "oct": "-1|0|1", "acc": '""|"#"|"b"', "conf": "0-1"}],
    "missed": [{"after_i": "int", "digit": "0-7", "oct": "-1|0|1", "acc": '""|"#"|"b"'}],
    "reject": ["int (draft index that is not a note)"],
    "verdict": "ok|fixed|uncertain",
    "reason": "string",
}


def _png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_request(row_id: str, crop_bgr: np.ndarray, draft: List[Dict], model: str = "qwen3-vl-30b-a3b") -> Dict:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f'row_id={row_id}\n草稿={json.dumps(draft, ensure_ascii=False)}\n'
                             f'请只输出符合 schema 的 JSON。字段取值说明：{json.dumps(RESPONSE_HINT, ensure_ascii=False)}'},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(crop_bgr)}},
                ],
            },
        ],
    }
    return payload


def _post(payload: Dict, base_url: str, api_key: str, timeout: int = 60) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf8"))


def call(payload: Dict, base_url: Optional[str] = None, api_key: Optional[str] = None) -> Optional[Dict]:
    base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    api_key = api_key or os.environ.get("LLM_API_KEY", "")
    if not (base_url and api_key):
        return None
    body = _post(payload, base_url, api_key)
    txt = body["choices"][0]["message"]["content"]
    txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(txt)


def dry_run(payload: Dict, tag: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.json")
    with open(path, "w", encoding="utf8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return path


def apply_edits(draft: List[Dict], answer: Dict) -> Tuple[List[Dict], List[str]]:
    """Apply the model's minimal edits; every change is logged with a reason."""
    out = [dict(t) for t in draft]
    log: List[str] = []
    for item in answer.get("notes", []):
        i = int(item["i"])
        if 0 <= i < len(out):
            before = dict(out[i])
            for k in ("digit", "oct", "acc"):
                if k in item and item[k] not in (None, ""):
                    out[i][k] = item[k] if k != "oct" else int(item[k])
            if out[i] != before:
                log.append(f"#{i}: {before['digit']}/{before['oct']}/{before['acc']} -> "
                           f"{out[i]['digit']}/{out[i]['oct']}/{out[i]['acc']}")
            out[i]["conf"] = 1.0
    for item in answer.get("missed", []):
        i = int(item.get("after_i", len(out) - 1))
        out.insert(i + 1, {"digit": item["digit"], "oct": int(item.get("oct", 0)),
                           "acc": item.get("acc", ""), "conf": 0.5, "inserted": True})
        log.append(f"插入 #{i+1}: {item['digit']} (模型认为漏识别)")
    for i in answer.get("reject", []):
        i = int(i)
        if 0 <= i < len(out):
            out[i]["rejected"] = True
            log.append(f"#{i}: 判为非音符，剔除")
    return out, log


def verifier_cascade(rows: List[Dict], crops: List[np.ndarray], tier: str = "cheap") -> None:
    """Documented cascade: only low-confidence rows escalate to the flagship model."""
    raise NotImplementedError("wire this to your provider; see docs/优化方案.md §5")


def estimate_cost(n_rows: int, tier: str = "cheap", rows_needing_llm: float = 0.35) -> Dict[str, float]:
    pin, pout = PRICES[tier]
    # one row crop ~0.6k vision tokens (1000x60 px @ 28px/token patch grid)
    tin_per_row, tout_per_row = 900, 220
    n = n_rows * rows_needing_llm
    usd = (n * tin_per_row * pin + n * tout_per_row * pout) / 1e6
    return {"rows": n_rows, "rows_to_llm": round(n, 1), "usd_per_1000_pages": round(usd * 1000 / max(1, n_rows / 12), 2)}


def mock_answer(draft: List[Dict], row_id: str) -> Dict:
    """Deterministic stand-in so the plumbing can be exercised without a key."""
    notes = []
    for i, t in enumerate(draft):
        if float(t.get("conf", 1.0)) < 0.62 and t.get("digit") in ("5", "6"):
            other = "6" if t["digit"] == "5" else "5"
            notes.append({"i": i, "digit": other, "reason": "低频置信度 5/6 混淆"})
    reject = [i for i, t in enumerate(draft) if t.get("digit") is None]
    return {"row_id": row_id, "notes": notes, "missed": [], "reject": reject,
            "verdict": "fixed" if notes or reject else "ok", "reason": "mock 审核"}


if __name__ == "__main__":
    draft = [
        {"digit": "6", "oct": 0, "acc": "", "conf": 0.93},
        {"digit": "5", "oct": 0, "acc": "", "conf": 0.44},
        {"digit": "3", "oct": 1, "acc": "", "conf": 0.88},
        {"digit": "0", "oct": 0, "acc": "", "conf": 0.29},
    ]
    img = np.full((60, 800, 3), 255, np.uint8)
    payload = build_request("r12", img, draft)
    p = dry_run(payload, "sample_row12")
    print("dry-run payload ->", p)
    ans = mock_answer(draft, "r12")
    fixed, log = apply_edits(draft, ans)
    print("edits:", log)
    print("after:", json.dumps(fixed, ensure_ascii=False))
    for tier in ("qwen3-vl-30b-a3b", "gemini-2.5-flash", "gpt-5"):
        print(tier, estimate_cost(12, tier))
