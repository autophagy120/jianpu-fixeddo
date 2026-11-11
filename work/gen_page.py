"""把 work\roi_page.html 注入 jp\roi_server.py 的 PAGE 常量。改完 UI 跑一次。"""
import io, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_p = os.path.join(ROOT, "work", "roi_page.html")
py_p = os.path.join(ROOT, "jp", "roi_server.py")

html = io.open(html_p, encoding="utf8").read()
src = io.open(py_p, encoding="utf8").read()
lines = src.split("\n")
n = 0
for i, ln in enumerate(lines):
    if ln.startswith("PAGE = "):
        lines[i] = "PAGE = " + repr(html)
        n += 1
if n != 1:
    sys.exit("PAGE 行数量异常: %d" % n)
io.open(py_p, "w", encoding="utf8", newline="\n").write("\n".join(lines))
print("已注入 PAGE：html %d 字符 -> %s" % (len(html), py_p))
