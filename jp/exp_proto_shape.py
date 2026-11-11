"""验证：1008 个原型是不是 5x6/6x5 的二值字形网格。"""
import numpy as np

T = np.load(r"D:\简谱转调项目\work\extracted\JianPu-Easy-Trans\templates.npy")
print("shape", T.shape)
print("第0行是否等于各列最大值:", np.allclose(T[0], T.max(axis=0)))
P = T[1:]                      # 1008 个原型
print("原型矩阵", P.shape, "取值范围", np.unique(P)[:6], "...")

# 转成严格二值
B = (P > 0.5).astype(np.int8)
uniq, counts = np.unique(B, axis=0, return_counts=True)
print(f"\n不同的原型图案 {len(uniq)} 种；出现次数分布 top10: {sorted(counts)[::-1][:10]}")

# 把最常见的几种画成 6x5 / 5x6
for shape in ((6, 5), (5, 6)):
    print(f"\n=== 按 {shape[0]}x{shape[1]} 显示最常见的 8 种 ===")
    order = np.argsort(-counts)[:8]
    for k in order:
        g = uniq[k].reshape(shape)
        n = int(counts[k])
        print(f"  (出现 {n} 次)")
        for row in g:
            print("   " + "".join("█" if v else "·" for v in row))
