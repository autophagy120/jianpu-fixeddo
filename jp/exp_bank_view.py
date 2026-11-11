import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from jp import templbank as TB, core

cs, labels, counts = TB.load()
order = np.argsort(-counts)[:60]
print("top60 计数:", counts[order][:20], "...")

tile, cols = 110, 10
rows = int(np.ceil(len(order) / cols))
canvas = np.full((rows * (tile + 40), cols * (tile + 12), 3), 255, np.uint8)
for k, idx in enumerate(order):
    r, c = divmod(k, cols)
    y0, x0 = r * (tile + 40) + 30, c * (tile + 12) + 6
    big = cv2.resize(255 - cs[idx] * 255, (tile, tile), interpolation=cv2.INTER_NEAREST)
    canvas[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + tile, y0 + tile), (180, 180, 180), 1)
    cv2.putText(canvas, f"{k}", (x0 + 30, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
core.imwrite("out/bank_top60.png", canvas)
print("written out/bank_top60.png", canvas.shape)
