"""可选的 LLM 辅助：只吃 **OCR 后的文字**，绝不上传图片。

为什么要这么设计（需求方明确）：
- 图片送进多模态接口，一页就是几千 token，成本用不起；
- 而 OCR 文本一页才 200–800 token，只在"补标题/歌手/调号"这种低频操作里用。

配置（任选其一，key 不会写进日志）：
- `work\\llm.json`：{"base": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key": "sk-..."}
- 环境变量：JIANPU_LLM_BASE / JIANPU_LLM_MODEL / JIANPU_LLM_KEY

只支持 OpenAI 兼容的 `/chat/completions`。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

CFG_FILES = [os.path.join("work", "llm.json"), os.path.join("work", "llm.local.json")]

# 默认就用 DeepSeek（2026-11 官方文档）：
#   BASE URL (OpenAI 格式) = https://api.deepseek.com
#   模型 = deepseek-flash（= DeepSeek-V4.1-Flash-1017，旧的 deepseek-v4-flash 仍兼容）
# 只要在「设置 → AI 配置」里补一个 key 就能用。
DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def config() -> dict:
    cfg = dict(base=os.environ.get("JIANPU_LLM_BASE", "") or DEFAULT_BASE,
               model=os.environ.get("JIANPU_LLM_MODEL", "") or DEFAULT_MODEL,
               key=os.environ.get("JIANPU_LLM_KEY", ""),
               use_search=True, workers=10)
    for p in CFG_FILES:
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf8"))
                for k in ("base", "model", "key"):
                    if d.get(k):
                        cfg[k] = d[k]
                if "use_search" in d:
                    cfg["use_search"] = bool(d["use_search"])
                if d.get("workers"):
                    cfg["workers"] = max(1, min(16, int(d["workers"])))
            except Exception:
                pass
    return cfg


def available() -> bool:
    c = config()
    return bool(c["base"] and c["key"] and c["model"])


def mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return key[:3] + "…" + key[-4:]


def public_config() -> dict:
    """给前端看的配置（**不含完整 key**）。"""
    c = config()
    src = "env" if os.environ.get("JIANPU_LLM_KEY") else ""
    for p in CFG_FILES:
        if os.path.exists(p):
            src = src or p
    return dict(configured=available(), base=c["base"], model=c["model"],
                has_key=bool(c["key"]), key_mask=mask(c["key"]), source=src,
                use_search=bool(c.get("use_search", True)),
                workers=int(c.get("workers", 10) or 10))


def save_config(base: str = "", model: str = "", key: str = "",
                use_search=None, workers=None) -> dict:
    """写进 work\\llm.json。key 留空 = 保持原来的（不会被清掉）。"""
    cur = config()
    data = dict(base=(base or cur["base"] or "").strip(),
                model=(model or cur["model"] or "").strip(),
                key=(key or cur["key"] or "").strip(),
                use_search=bool(cur.get("use_search", True) if use_search is None else use_search),
                workers=int(cur.get("workers", 10) if workers is None else workers))
    p = CFG_FILES[0]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return public_config()


def test_connection(max_tokens: int = 24) -> dict:
    """发一次最小请求验证 base/model/key 能不能通（只发一句 'ping'，不涉及谱面）。

    DeepSeek 默认是「思考模式」；这里显式关掉，否则小 max_tokens 会被 reasoning
    吃光、content 返回空（关掉后能真正看到字，也省 token）。
    """
    c = config()
    if not (c["base"] and c["key"] and c["model"]):
        return dict(ok=False, error="配置不完整（base / model / key 都要有）")
    url = c["base"].rstrip("/") + "/chat/completions"
    try:
        r = _post(url, dict(model=c["model"], max_tokens=max_tokens,
                            thinking=dict(type="disabled"),
                            messages=[dict(role="user", content="ping")]), c["key"], timeout=40)
    except Exception as e:
        return dict(ok=False, error="%s: %s" % (type(e).__name__, str(e)[:160]))
    txt = ""
    try:
        txt = (r["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        pass
    usage = r.get("usage") or {}
    return dict(ok=True, sample=txt[:60] or "(空回复)",
                usage=dict(prompt=usage.get("prompt_tokens"),
                           completion=usage.get("completion_tokens")),
                model=r.get("model") or c["model"])


def _post(url: str, body: dict, key: str, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf8"))


def _first_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


PROMPT_META = (
    "你是乐谱整理助手。下面是一首简谱经 OCR 得到的文字（含乱码，可能混有歌词）。"
    "请只输出一个 JSON 对象，字段："
    '{"title": 曲名, "artist": 歌手或作者, "key": 调号如 "G" 或 "bA"（谱头写的 1=?，读不到就给空串）}。'
    "不要输出任何解释，不要编造读不出来的内容，读不到就留空串。"
)

PROMPT_ID_SEARCH = (
    "你在根据「歌词片段 + 网页搜索结果」判断这份简谱是哪首歌。"
    "规则：\n"
    "1) 曲名**只能**从搜索结果的标题或摘要里出现过的名称中选，不允许自己编；\n"
    "2) 如果搜索结果里没有明确对得上的歌名，title 必须为空字符串；\n"
    "3) 宁可空着也不要猜。\n"
    "只输出 JSON："
    '{"title":"","artist":"","key":"","confidence":"high|low","evidence":"引用的结果标题"}'
)


# ---------------------------------------------------------------- 联网搜索
def search(query: str, count: int = 5, need_content: bool = False, timeout: int = 40) -> list:
    """DeepSeek 官方 Search API：POST /v1/tools/search。"""
    c = config()
    base = c["base"].rstrip("/")
    url = base + ("/tools/search" if base.endswith("/v1") else "/v1/tools/search")
    body = dict(query=query[:128] or "简谱", count=max(1, min(50, count)),
                need_content=bool(need_content))
    r = _post(url, body, c["key"], timeout=timeout)
    return r.get("results") or []


def lyric_snippet(txt: str, max_chars: int = 36) -> str:
    """从 OCR 文本里挑一句最像歌词的（假名/汉字多、数字少），用来搜歌名。"""
    import re as _re
    best, best_score = "", 0
    for line in (txt or "").splitlines():
        line = line.strip()
        if len(line) < 4:
            continue
        # 跳过谱头/调号/纯数字行
        if _re.search(r"1\s*=|4/4|3/4|6/8", line):
            continue
        cjk = sum(1 for ch in line if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
        digits = sum(1 for ch in line if ch.isdigit())
        if cjk < 4 or digits > len(line) * 0.2:
            continue
        score = cjk - digits * 2
        if score > best_score:
            best, best_score = line, score
    clean = _re.sub(r"[，。！？、,.!?;；:：\"'“”‘’()（）\[\]【】\s]+", " ", best).strip()
    return clean[:max_chars]


def title_candidates(txt: str, limit: int = 3) -> list:
    """从 OCR 文本里挑可能的**曲名行**（通常是前几行，短、没有数字和调号）。"""
    import re as _re
    out = []
    for ln in (txt or "").splitlines():
        s0 = ln.strip()
        if not s0 or len(s0) > 24:
            continue
        if _re.search(r"1\s*=|\d", s0):
            continue
        cjk = sum(1 for ch in s0 if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff"
                  or ch.isalpha())
        if cjk < 2:
            continue
        s0 = _re.sub(r"^[\[【(（\s·\-—:：]+|[\]】)）\s·\-—:：]+$", "", s0)
        if s0 and s0 not in out:
            out.append(s0)
        if len(out) >= limit:
            break
    return out


def identify(ocr_text: str, use_search: bool = True, max_chars: int = 1200) -> dict:
    """识别这份谱是哪首歌。

    有联网时：先用**歌词片段**搜网页 → 把搜索结果喂给模型，只允许从结果里选歌名；
    模型给不出证据/证据对不上 → 返回空标题（宁可不标，也不乱标）。
    """
    c = config()
    if not available():
        raise RuntimeError("未配置 AI：请在 设置 → AI 配置 里填写 base/model/key")
    txt = (ocr_text or "")[:max_chars]
    if not txt.strip():
        return {}
    snip = lyric_snippet(txt)
    used_search, results = False, []
    if use_search and snip:
        try:
            results = search(snip + " 歌词", count=5)
            used_search = True
        except Exception:
            used_search = False
    if used_search and results:
        ctx = "\n".join("- 标题：%s\n  摘要：%s" % (r.get("title") or "",
                                                   (r.get("snippet") or r.get("summary") or "")[:140])
                        for r in results[:5])
        user = "歌词片段：%s\n\n搜索结果：\n%s\n\n谱面 OCR（截断）：\n%s" % (snip, ctx, txt)
        prompt = PROMPT_ID_SEARCH
    else:
        user = txt
        prompt = PROMPT_META

    url = c["base"].rstrip("/") + "/chat/completions"
    body = dict(model=c["model"], max_tokens=400,
                thinking=dict(type="disabled"),
                response_format=dict(type="json_object"),
                messages=[dict(role="system", content=prompt),
                          dict(role="user", content=user)])
    r = _post(url, body, c["key"])
    try:
        out = r["choices"][0]["message"]["content"] or ""
    except Exception:
        return {}
    d = _first_json(out)
    res = {k: str(v or "").strip() for k, v in d.items()
           if k in ("title", "artist", "key", "confidence", "evidence")}
    res["search"] = bool(used_search and results)
    res["hits"] = [r0.get("title") for r0 in results[:5]] if used_search else []
    # 反幻觉：联网模式下，标题必须在搜索结果里出现过，否则丢弃
    if res["search"] and res.get("title"):
        blob = " ".join(((r0.get("title") or "") + " " + (r0.get("snippet") or "")
                         + " " + (r0.get("summary") or "")) for r0 in results).lower()
        core = res["title"].lower()
        if core not in blob:
            res["title"] = ""
            res["confidence"] = "low"
    if used_search and not res.get("title"):
        # 歌词搜不到 → 用 OCR 的标题候选行再搜（第1行常是曲名），
        # 只有搜索结果里真出现这个字符串才采纳，避免瞎标
        for cand in title_candidates(txt)[:2]:
            try:
                rs2 = search(cand + " 歌", count=3)
            except Exception:
                continue
            blob2 = " ".join(((r0.get("title") or "") + " " + (r0.get("snippet") or "")
                              + " " + (r0.get("summary") or "")) for r0 in rs2).lower()
            if cand.lower() in blob2:
                res["title"] = cand
                res["confidence"] = "medium"
                res["evidence"] = "OCR 首行 + 搜索结果"
                res["hits"] = [r0.get("title") for r0 in rs2[:3]]
                break
    if used_search and not res.get("title"):
        res["confidence"] = res.get("confidence") or "low"
    return res


def extract_meta(ocr_text: str, max_chars: int = 1500) -> dict:
    """用 OCR 文本猜 曲名/歌手/调号。**不含任何图片数据**。"""
    c = config()
    if not available():
        raise RuntimeError("未配置 LLM（work\\llm.json 或 JIANPU_LLM_* 环境变量）")
    text = (ocr_text or "")[:max_chars]
    if not text.strip():
        return {}
    url = c["base"].rstrip("/") + "/chat/completions"
    body = dict(model=c["model"], max_tokens=400,
                thinking=dict(type="disabled"),          # 抽取任务不需要思考，省钱且更稳
                response_format=dict(type="json_object"),  # 要求直接给 JSON
                messages=[dict(role="system", content=PROMPT_META),
                          dict(role="user", content=text)])
    r = _post(url, body, c["key"])
    try:
        out = r["choices"][0]["message"]["content"] or ""
    except Exception:
        return {}
    d = _first_json(out)
    return {k: str(v or "").strip() for k, v in d.items() if k in ("title", "artist", "key")}
