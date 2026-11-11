"""All tunables in one JSON file, so nothing is hard-coded in the CLI.

    python -m jp.settings            # write ./settings.json with defaults and print it
    python -m jp.run ... --settings settings.json --set harmonica.breath_penalty=1.5

Precedence: defaults < settings.json < --set k=v < explicit CLI flags.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PATH = os.path.join(ROOT, "settings.json")

DEFAULTS: Dict[str, Any] = {
    "_说明": "改成你想要的即可；命令行 --set 段.键=值 可临时覆盖",

    "input": {
        "exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"],
        "skip_produced": True,          # 跳过 easytr 已生成的 “(原调X-转调Y)” 文件
        "work_digit_height": 28,        # 归一化到多少像素高的数字再识别
        "min_digit_height": 10,         # 低于这个像素高就提示"分辨率不足，可能不准"
    },

    "recognize": {
        "digit_min_prob": 0.55,         # 低于此置信度视为"不确定"，进待核对队列
        "acc_min_prob": 0.60,           # 升降号的分类阈值
        "glyph_min_height_ratio": 0.45, # 相对字高的最小"字形"高度
        "dot_max_height_ratio": 0.30,   # 八度点的最大高度（相对字高）
        "dot_max_gap_ratio": 1.00,      # 八度点离数字的最大间距（相对字高）
        "rule_hlen_ratio": 0.75,        # 去减时线的核长（相对字高）
    },

    "layout": {
        "music_row_digit_frac": 0.28,   # 一行里数字占比超过多少算谱行
        "music_row_min_glyphs": 4,
        "lyric_row_max_digit_frac": 0.28,
        "absorb_mark_rows": True,       # 把"只有点"的窄带并回相邻谱行
    },

    "key": {
        "auto_from_filename": True,     # 从 “(原调#G-转调C)” 里读原调
        "auto_from_sibling": True,      # 原调只在姊妹文件名里时，同目录找
        "declared_in_image": True,      # 尝试从谱面 “1=bB” 读（未实现则回退到 --key）
    },

    "transpose": {
        "target": "C",                  # 转调图的目标调
        "prefer_flat_spelling": False,  # 等音并列时优先降号（图输出一般用升号）
        "octave_mode": "absolute",      # absolute(绝对音高,默认) | keep(八度点原样) | wrap(级数进位)
    },

    "fixed_do": {
        "ref": "auto",                  # auto | c | csharp
        "csharp_threshold": 3,          # C 记谱超过几个升降号就切 C#
        "prefer_flat": True,            # C# 记谱时"全用降号"
        "style": "cn",                  # cn: （1）【1】   ascii: (1)[1]
    },

    "harmonica": {
        "enabled": True,
        "breath_penalty": 1.0,          # 换气方向（吹↔吸）代价
        "slide_penalty": 0.6,           # 推键切换代价
        "accidental_penalty": 0.6,      # 每多一个升降号的可读性代价
        "allow_degree_swap": True,      # 允许 #3<->4、b1<->7 这类跨音级等音
        "annotate": False,              # 音符后标 (吹)(吸)(推)
    },

    "phrases": {
        "mode": "punctuation",          # punctuation | bars | line
        "bars_per_phrase": 4,
    },

    "lyrics": {
        "mode": "align",                # align | group | inline | tsv | none
        "source": "none",               # none | file | api
        "file": "",                     # 每行一句的歌词文本
        "continuation": "～",           # 拖腔占位符（align 模式）
        "max_notes_per_char": 4,        # 一个字最多分到几个音
        "align_anchors": True,          # 用标点/小节线做对齐锚点
    },

    "output": {
        "formats": ["txt", "img", "json"],
        "font": r"C:\Windows\Fonts\msyh.ttc",
        "ink_color": "red",          # 重绘数字的颜色：red|blue|green|black|#RRGGBB（认出=彩色，没认出=原图黑色）
        "footer": True,
        "orig_key_name_style": "flat",  # 页脚里原调写法 flat|sharp
    },

    "llm": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "max_tokens": 2000,
        "use_for": ["lyric_ocr", "row_check"],   # 打算用大模型做的事
    },
}


def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load(path: str = "") -> Dict[str, Any]:
    path = path or DEFAULT_PATH
    cfg = json.loads(json.dumps(DEFAULTS))     # deep copy
    if os.path.exists(path):
        with open(path, encoding="utf8") as fh:
            _merge(cfg, json.load(fh))
    return cfg


def save(cfg: Dict[str, Any], path: str = "") -> str:
    path = path or DEFAULT_PATH
    with open(path, "w", encoding="utf8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    return path


def apply_overrides(cfg: Dict[str, Any], pairs: List[str]) -> Dict[str, Any]:
    """--set harmonica.breath_penalty=1.5"""
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set 需要 段.键=值 形式，收到 {p!r}")
        k, v = p.split("=", 1)
        node = cfg
        parts = k.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise SystemExit(f"--set {k}: {part} 不是字典")
        cur = node.get(parts[-1])
        if isinstance(cur, bool):
            node[parts[-1]] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            node[parts[-1]] = int(float(v))
        elif isinstance(cur, float):
            node[parts[-1]] = float(v)
        elif isinstance(cur, list):
            node[parts[-1]] = [x.strip() for x in v.split(",") if x.strip()]
        else:
            node[parts[-1]] = v
    return cfg


def flat(cfg: Dict[str, Any], prefix: str = "") -> List[str]:
    out = []
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out += flat(v, key + ".")
        else:
            out.append(f"{key} = {v!r}")
    return out


def __main__():
    cfg = load()
    p = save(cfg)
    print(f"写出默认配置 {p}")
    print("\n".join(flat(cfg)))


if __name__ == "__main__":
    __main__()
