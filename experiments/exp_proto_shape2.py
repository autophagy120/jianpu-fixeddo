import numpy as np

T = np.load(r"D:\简谱转调项目\work\extracted\JianPu-Easy-Trans\templates.npy")
P = T[1:]
B = (P > 0.5).astype(np.int8)
uniq, counts = np.unique(B, axis=0, return_counts=True)
order = np.argsort(-counts)
print("最前者出现次数:", counts[order[:5]])
print("每行 0/1 之外的列统计：非二值元素总数 =", int(((P != 0) & (P != 1)).sum()))

for shape in ((3, 10), (10, 3), (5, 6), (6, 5), (2, 15), (15, 2)):
    print(f"\n=== {shape[0]}x{shape[1]} : 出现最多的 4 种 ===")
    for k in order[:4]:
        g = uniq[k].reshape(shape)
        print(f"  ({int(counts[k])} 次)")
        for row in g:
            print("    " + "".join("█" if v else "·" for v in row))

# 非二值元素都在哪些列
mask = (P != 0) & (P != 1)
cols = mask.sum(axis=0)
print("\n非二值元素按列:", {i: int(c) for i, c in enumerate(cols) if c})
rows = mask.sum(axis=1)
print("含非二值元素的行数:", int((rows > 0).sum()), "其中最多的行:", np.argsort(-rows)[:5])
