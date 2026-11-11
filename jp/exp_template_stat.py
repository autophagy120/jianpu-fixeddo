"""反推 easytr 的 30 维特征：先看 templates.npy 的数值结构。"""
import numpy as np

T = np.load(r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\..\..\..\..\..\DeepSeek-Harness\data\attachments\v1\files\42\42063761d777ae3cb3a722a145d8797bee5d0967c0cf14bb677f852da65e71cb\JianPu-Easy-Trans.rar", allow_pickle=False) if False else None

path = r"D:\简谱转调项目\work\extracted\JianPu-Easy-Trans\templates.npy"
T = np.load(path)
print("shape", T.shape, "dtype", T.dtype)
print("min/max", T.min(), T.max())

np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("\n每列(min, max, mean, 唯一值数):")
for j in range(T.shape[1]):
    col = T[:, j]
    print(f"  dim{j:02d}  min={col.min():8.3f}  max={col.max():8.3f}  mean={col.mean():7.3f}  "
          f"uniq={len(np.unique(col)):4d}  整数比例={np.mean(np.isclose(col, np.round(col))):.2f}")

print("\n前 3 个原型:")
print(T[:3])

print("\n是否有负数/是否像被中心化:")
neg = (T < 0).sum()
print("  负值个数", neg, " 占", round(neg / T.size, 5))
print("  全局均值", T.mean())
