"""用预训练 MNIST 模型（ONNX）识别小字号印刷数字。

MNIST 官方预处理：先等比缩放到 20x20 的内容框，再按**质心**居中放进 28x28。
我的 CNN 之前用的是"保持宽高比 + 居中填充"，对小字（6-9 px）不匹配；
预训练模型配官方预处理往往直接就对了。
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_ONNX = os.path.join(ROOT, "models", "mnist-12.onnx")


def mnist_preprocess(patch: np.ndarray) -> np.ndarray:
    """ink=白(255) 的二值小块 -> MNIST 约定 (1,1,28,28) float32 [0,1]"""
    a = patch
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return np.zeros((1, 1, 28, 28), np.float32)
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
    h, w = sub.shape
    sc = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
    sub = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.float32)
    # 按质心居中
    ys2, xs2 = np.nonzero(sub > 0.2)
    if len(ys2):
        cy, cx = ys2.mean(), xs2.mean()
    else:
        cy, cx = nh / 2.0, nw / 2.0
    oy = int(round(14 - cy))
    ox = int(round(14 - cx))
    oy = max(0, min(28 - nh, oy))
    ox = max(0, min(28 - nw, ox))
    canvas[oy:oy + nh, ox:ox + nw] = sub
    return canvas[None, None, :, :]


class MnistNet:
    def __init__(self, path: str = DEFAULT_ONNX):
        import onnxruntime as ort

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def predict_batch(self, patches: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not patches:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        labels, confs = [], []
        for p in patches:                      # 该模型的 batch 维固定为 1
            x = mnist_preprocess(p)
            out = self.sess.run(None, {self.iname: x})[0].reshape(-1)
            e = np.exp(out - out.max())
            prob = e / e.sum()
            labels.append(int(prob.argmax()))
            confs.append(float(prob.max()))
        return np.array(labels, dtype=np.int64), np.array(confs, dtype=np.float32)
