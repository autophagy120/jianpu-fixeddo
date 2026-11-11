import os
import sys
import traceback

sys.path.insert(0, r"D:\简谱转调项目")
import jp.library as L
from jp.roi_server import State

it = L.get("ce9220b97890")
txt = it.get("ocr_txt")
print("ocr_txt:", txt, os.path.exists(txt), "size:", os.path.getsize(txt))
try:
    st = State(it["image"], txt, "C", title=it.get("title") or "", cache_key=it.get("id"))
    print("OK pairs=%d uncovered=%d" % (len(st.pairs), len(st.uncovered)))
except Exception:
    traceback.print_exc()
