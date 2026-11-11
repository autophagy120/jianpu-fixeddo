"""Batched inference wrapper around the synthetic glyph CNN."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import synth

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CKPT = os.path.join(ROOT, "models", "glyph_cnn.pt")


class GlyphNet:
    def __init__(self, ckpt: str = DEFAULT_CKPT):
        import torch

        self.torch = torch
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"{ckpt} missing - run `python -m jp.synth` first")
        blob = torch.load(ckpt, map_location="cpu")
        self.classes: List[str] = blob["classes"]
        self.model = synth.build_model()
        self.model.load_state_dict(blob["state"])
        self.model.eval()
        self.digit_idx = [i for i, c in enumerate(self.classes) if c in "01234567"]

    def predict(self, patches: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """patches: list of uint8/float arrays (any size, ink=white) -> (cls_idx, prob)."""
        if not patches:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        batch = np.stack([synth.normalize(p.astype(np.uint8) if p.dtype != np.uint8 else p) for p in patches])
        with self.torch.no_grad():
            t = self.torch.from_numpy(batch)[:, None, :, :]
            logits = self.model(t)
            prob = self.torch.softmax(logits, dim=1).numpy()
        return prob.argmax(1), prob.max(1)

    def digit_scores(self, patch: np.ndarray) -> np.ndarray:
        """只在这 8 个类别里取概率，供"闭集兜底"用。"""
        batch = synth.normalize(patch.astype(np.uint8) if patch.dtype != np.uint8 else patch)
        with self.torch.no_grad():
            t = self.torch.from_numpy(batch)[None, None, :, :]
            prob = self.torch.softmax(self.model(t), dim=1).numpy()[0]
        return np.array([prob[self.classes.index(c)] for c in "01234567"], dtype=np.float32)

    def digit_with_reject(self, patches: List[np.ndarray], min_prob: float = 0.55) -> Tuple[List[Optional[str]], np.ndarray]:
        """Only return 0-7 when the model is confident AND thinks it is a digit."""
        cls, conf = self.predict(patches)
        out: List[Optional[str]] = []
        for i, c in enumerate(cls):
            name = self.classes[int(c)]
            out.append(name if (name in "01234567" and conf[i] >= min_prob) else None)
        return out, conf
