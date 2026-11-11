"""为 easytr 的 Cython 模块补上 scipy>=1.3 已删除的 scipy.misc.imresize。

getLeter / reco_note 在 import 时执行 `from scipy.misc import imresize`，
在 scipy 1.5.4 下会 ImportError。这里用 PIL 实现等价函数并注入 scipy.misc，
必须在 import 那些模块**之前**执行（本文件即为此设计，import 它即可）。
"""
import numpy as np
from PIL import Image


def _imresize(arr, size, interp="bilinear", mode=None):
    a = np.asarray(arr)
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        size = (int(a.shape[0] * size / 100.0 + 0.5), int(a.shape[1] * size / 100.0 + 0.5))
    h, w = int(size[0]), int(size[1])
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    resample = Image.BILINEAR if interp in ("bilinear", "bicubic", "lanczos") else Image.NEAREST
    return np.array(img.resize((w, h), resample))


def install():
    import scipy.misc
    scipy.misc.imresize = _imresize
    return True


install()
