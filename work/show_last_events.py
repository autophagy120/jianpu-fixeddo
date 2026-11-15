"""看最近的修正记录：谁把 5 改成了 4。"""
import io
import json
import os
import sys

sys.path.insert(0, r"D:\简谱转调项目")

for path in (r"D:\简谱转调项目\work\train\events.jsonl",
             r"D:\简谱转调项目\out\roi_server.log"):
    print("=" * 20, path)
    if not os.path.exists(path):
        print("(不存在)")
        continue
    lines = io.open(path, encoding="utf8", errors="replace").read().splitlines()
    for ln in lines[-12:]:
        print("  " + ln[:220])
