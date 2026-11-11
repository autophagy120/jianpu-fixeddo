# -*- coding: utf-8 -*-
"""完整提取 easytr 五个 Cython 模块的元信息。

Cython 用 binding=True 编译时，函数会带一个**伪造的 code object**；
虽然 co_code 不是真字节码，但 co_consts / co_names / co_varnames 往往
保留了原始 Python 源码里的常量、属性名、字典键、阈值和调名列表 ——
这就是"反编译"能拿到的最实质信息，比逐个试探参数强得多。

输出：work\et_decompile\<module>.txt（全部内容）
      屏幕：摘要（每个函数的参数名 + 有价值的常量）
"""
import io
import os
import sys
import types

OUT = r"D:\简谱转调项目\work\et_decompile"
os.makedirs(OUT, exist_ok=True)
APP = r"D:\简谱转调项目\work\extracted\JianPu-Easy-Trans"

sys.path.insert(0, r"D:\简谱转调项目\work")
sys.path.insert(0, APP)
os.chdir(APP)

import et_shim  # noqa: F401  (注入 scipy.misc.imresize)

MODULES = ["getLeter", "read_letter", "reco_note", "noteTranslate",
           "replaceimgTone", "ModifyNotes"]


def fmt(obj, maxlen=300):
    r = repr(obj)
    return r if len(r) <= maxlen else r[:maxlen] + "..."


def dump(mod_name):
    lines = []
    try:
        mod = __import__(mod_name)
    except Exception as e:
        return None, ["IMPORT FAIL: %s: %s" % (type(e).__name__, e)]
    lines.append("=" * 70)
    lines.append("MODULE %s   file=%s" % (mod_name, getattr(mod, "__file__", "?")))
    lines.append("  __doc__ = %s" % fmt(getattr(mod, "__doc__", None)))
    lines.append("  members: %s" % [a for a in dir(mod) if not a.startswith("_")])
    interesting = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, types.FunctionType):
            code = getattr(obj, "__code__", None)
            lines.append("-" * 60)
            lines.append("FUNCTION %s" % name)
            lines.append("  doc = %s" % fmt(getattr(obj, "__doc__", None)))
            if code is not None:
                cv = code.co_varnames
                lines.append("  co_varnames = %s" % (cv,))
                # 前 n 个是形参
                try:
                    narg = code.co_argcount
                except Exception:
                    narg = len(cv)
                lines.append("  ARGS(%d) = %s" % (narg, cv[:narg]))
                lines.append("  LOCALS = %s" % [v for v in cv[narg:]])
                lines.append("  co_names = %s" % (code.co_names,))
                cn = code.co_consts
                lines.append("  co_consts (%d):" % len(cn,))
                for i, c in enumerate(cn):
                    s = fmt(c, 200)
                    lines.append("    [%02d] %-12s %s" % (i, type(c).__name__, s))
                    if isinstance(c, (list, tuple, dict)) and len(c) >= 3:
                        interesting.append((name, "const_%d" % i, s))
                    if isinstance(c, str) and len(c) >= 3:
                        interesting.append((name, "str_%d" % i, s))
                lines.append("  co_code len = %d" % len(code.co_code or b""))
        elif isinstance(obj, (list, tuple, dict)) and len(obj) >= 3:
            lines.append("DATA %s = %s" % (name, fmt(obj, 600)))
            interesting.append((name, "data", fmt(obj, 300)))
        else:
            lines.append("OTHER %s : %s = %s" % (name, type(obj).__name__, fmt(obj, 200)))
    path = os.path.join(OUT, mod_name + ".txt")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(u"\n".join(lines))
    return path, interesting


all_interesting = {}
for m in MODULES:
    path, inter = dump(m)
    if path is None:
        print("[SKIP] %s: %s" % (m, inter[0]))
        continue
    print("[OK] %s -> %s" % (m, path))
    all_interesting[m] = inter

print("\n" + "=" * 70)
print("有价值的常量/数据（可能含调名列表、阈值、字典键）")
print("=" * 70)
for m, items in all_interesting.items():
    if not items:
        continue
    print("\n## " + m)
    seen = set()
    for fn, kind, s in items:
        key = (fn, s[:80])
        if key in seen:
            continue
        seen.add(key)
        print("  %-22s %-10s %s" % (fn, kind, s))
