"""探一下本机 cv2 / PIL 到底能解哪些图片格式，决定导入要不要转码。"""
import numpy as np
import cv2
from PIL import Image

FORMATS = [
    ("jpg", True), ("png", True), ("gif", True), ("bmp", True),
    ("webp", True), ("tif", True), ("avif", False), ("heic", False),
]

# 造一张小图，用 PIL 存成各格式，再用 cv2 / PIL 读回来
base = Image.new("RGB", (64, 48), (255, 255, 255))
for x in range(64):
    for y in range(48):
        if (x // 8 + y // 8) % 2 == 0:
            base.putpixel((x, y), (0, 0, 0))

print("%-6s %-8s %-8s %-8s" % ("格式", "PIL存", "cv2解", "PIL解"))
for ext, _ in FORMATS:
    p = "out\\_fmt_test." + ext
    try:
        if ext in ("avif", "heic"):
            raise RuntimeError("PIL 不支持写")
        base.save(p)
        wrote = True
    except Exception as e:
        print("%-6s %-8s %-8s %-8s  (%s)" % (ext, "✗", "-", "-", str(e)[:28]))
        continue
    raw = np.fromfile(p, np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    try:
        im = Image.open(p)
        im.load()
        pil_ok = True
    except Exception:
        pil_ok = False
    print("%-6s %-8s %-8s %-8s" % (ext, "✓", "✓" if bgr is not None else "✗",
                                   "✓" if pil_ok else "✗"))

print("\ncv2 支持的扩展名（部分）：")
for e in ("jpg", "jpeg", "png", "bmp", "gif", "webp", "tif", "tiff", "avif", "heic"):
    try:
        base.save("out\\_fmt_test2." + e)
        ok = cv2.imdecode(np.fromfile("out\\_fmt_test2." + e, np.uint8), cv2.IMREAD_COLOR) is not None
        print("  %-5s %s" % (e, "读得到" if ok else "读不到"))
    except Exception as ex:
        print("  %-5s 存不了 (%s)" % (e, str(ex)[:30]))
