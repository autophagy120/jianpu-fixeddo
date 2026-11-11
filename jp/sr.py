"""SRCNN 3x 超分（OrpheusNet 里内置的 Mirwaisse/SRCNN 权重）。

用途
----
把低分辨率扫描的简谱页（例如数字只有 9 px 高）整体放大 3 倍。这是**复原式**超分：
先把图 bicubic 放大到目标尺寸，再用 CNN 修正 bicubic 带来的模糊/振铃，所以输出比
纯 bicubic 更锐利，但不会凭空补出笔画结构。

与 notebook / 官方代码的一致性
------------------------------
`SRCNN(zoom=3).ipynb` 本身只有 shell 单元（`git clone Mirwaisse/SRCNN` +
`python main.py/run.py`），没有任何模型代码；模型定义与推理流程取自
https://github.com/Mirwaisse/SRCNN 的 `model.py` / `run.py`，并用权重的
state_dict 反验（形状完全吻合）：

    conv1: Conv2d(1, 64, 9, padding=4) + ReLU
    conv2: Conv2d(64, 32, 1, padding=0) + ReLU
    conv3: Conv2d(32, 1, 5, padding=2)            # 无激活
    参数量 8129

推理/预处理（严格照 `run.py`）：
  1. BGR -> RGB -> PIL YCbCr；
  2. **整幅** YCbCr 用 PIL BICUBIC 放大 zoom 倍（这也是"预放大"）；
  3. 只取 Y 通道 -> float32 / 255.0 -> (1,1,H,W)；
  4. 网络输出仍是 [0,1]，*255 后 clip(0,255) -> uint8；
  5. 与（已 bicubic 放大的）Cb/Cr 合并回 YCbCr -> RGB -> BGR。
即：**只对亮度 Y 做增强，色度通道不经过网络**。

权重文件说明
------------
`pths/SRCNN(zoom=3).pth` 不是裸 state_dict，而是 `torch.save(model)` 存下来的
**整个 nn.Module**（pickle 里是 `GLOBAL 'model SRCNN'`），所以 `torch.load` 需要
`weights_only=False` 并临时注入一个名为 ``model`` 的影子模块；载入后取 state_dict，
再 `strict=True` 装进本模块自己重新实现的 SRCNN。加载全程只做一次（模块级懒加载 + 锁）。

路径注意：`cv2.imread/imwrite` 不支持中文路径，请用 :func:`read_bgr` / :func:`write_bgr`
（或 jp.core.load_bgr / jp.core.imwrite），它们走 `np.fromfile` + `cv2.imdecode`。
"""
from __future__ import annotations

import os
import sys
import threading
import types
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 只在真正要用的时候才 import torch（导入 jp.sr 本身很便宜）
_torch = None  # type: ignore[assignment]

ZOOM = 3                      # 权重对应的放大倍数
TILE = 448                    # 分块推理的块边长（纯为省内存，结果与整图等价）
HALO = 8                      # 块间重叠：感受野半径 4+0+2=6，取 8 更保险
_N_THREADS = 4                # torch 线程数上限（避免和小米 OCR 等抢满 12 核）

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEIGHT_CANDIDATES: Tuple[str, ...] = (
    os.path.join(_ROOT, "work", "ref", "orpheus_repo", "OrpheusNet-main", "pths", "SRCNN(zoom=3).pth"),
    os.path.join(_ROOT, "work", "ref", "OrpheusNet-main", "pths", "SRCNN(zoom=3).pth"),
    os.path.join(_ROOT, "pths", "SRCNN(zoom=3).pth"),
)

_lock = threading.Lock()
_cache: Dict[str, object] = {"loaded": False, "model": None, "path": None, "err": None}
LAST: Dict[str, object] = {}   # 最近一次 upscale 的统计信息


# --------------------------------------------------------------------- io
def read_bgr(path: str) -> np.ndarray:
    """读图（支持中文/空格路径，等价于 jp.core.load_bgr）。"""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_bgr(path: str, img: np.ndarray, quality: Optional[int] = None) -> None:
    """写图（支持中文/空格路径，等价于 jp.core.imwrite）。"""
    ext = os.path.splitext(path)[1] or ".png"
    params: List[int] = []
    if quality is not None and ext.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buf = cv2.imencode(ext, img, params) if params else cv2.imencode(ext, img)
    if not ok:
        raise IOError(path)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    buf.tofile(path)


# ------------------------------------------------------------------ model
def weight_path() -> Optional[str]:
    """权重文件路径：环境变量 JP_SRCNN_PTH 优先，否则按默认位置找。"""
    env = os.environ.get("JP_SRCNN_PTH")
    if env:
        return env if os.path.isfile(env) else None
    for p in _WEIGHT_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def _build_class():
    torch = _torch
    nn = torch.nn

    class SRCNN(nn.Module):
        """与 Mirwaisse/SRCNN `model.py` 逐层一致的 3 层 SRCNN（1 通道 Y）。"""

        def __init__(self) -> None:          # noqa: D107
            super().__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
            self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        def forward(self, x):                # noqa: D102
            F = torch.nn.functional
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            return self.conv3(out)

    return SRCNN


def _extract_state_dict(path: str) -> Dict[str, object]:
    """把 .pth 读成 state_dict，兼容「整模块 pickle」和「裸 state_dict」两种存法。"""
    torch = _torch
    # 1) 正常情况：裸 state_dict
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and obj and all(hasattr(v, "shape") for v in obj.values()):
            return dict(obj)
    except Exception:
        pass
    # 2) 整个 nn.Module 被 pickle 了（本权重就是这种），需要假的 `model` 模块顶住
    shim = types.ModuleType("model")

    class _Shim(torch.nn.Module):
        def __init__(self, *a, **k):         # noqa: D107
            super().__init__()

    shim.SRCNN = _Shim
    had = sys.modules.get("model")
    sys.modules["model"] = shim
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if had is not None:
            sys.modules["model"] = had
        else:
            sys.modules.pop("model", None)
    if hasattr(obj, "state_dict"):
        return dict(obj.state_dict())
    if isinstance(obj, dict) and "model" in obj and hasattr(obj["model"], "state_dict"):
        return dict(obj["model"].state_dict())
    if isinstance(obj, dict):
        return {
            k: v for k, v in obj.items() if hasattr(v, "shape")
        }
    raise RuntimeError("无法从 %s 解析出 state_dict" % (path,))


def _load_model():
    """真正的加载逻辑；成功返回 eval 后的模型，失败抛异常。"""
    global _torch
    if _torch is None:
        import torch  # 延迟导入

        torch.set_num_threads(max(1, min(_N_THREADS, os.cpu_count() or 1)))
        _torch = torch
    torch = _torch

    path = weight_path()
    if path is None:
        raise FileNotFoundError(
            "找不到 SRCNN 权重，可设置环境变量 JP_SRCNN_PTH 指向 SRCNN(zoom=3).pth"
        )
    sd = _extract_state_dict(path)
    net = _build_class()()
    net.load_state_dict(sd, strict=True)      # 必须全 key 命中
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    _cache["path"] = path
    return net


def _get_model():
    """模块级懒加载 + 缓存（线程安全）。失败返回 None 并记录原因。"""
    if _cache["loaded"]:
        return _cache["model"]
    with _lock:
        if not _cache["loaded"]:
            try:
                _cache["model"] = _load_model()
                _cache["err"] = None
            except Exception as exc:          # noqa: BLE001 — 失败要能被 available() 吃掉
                _cache["model"] = None
                _cache["err"] = "%s: %s" % (type(exc).__name__, exc)
            _cache["loaded"] = True
    return _cache["model"]


def available() -> bool:
    """权重存在且模型能 strict 加载成功。第一次调用会真加载，之后走缓存。"""
    return _get_model() is not None


def last_error() -> Optional[str]:
    """加载/推理失败时的原因（调试用）。"""
    return _cache.get("err")  # type: ignore[return-value]


def describe() -> Dict[str, object]:
    """网络结构/参数量/权重路径，供报告与自检使用。"""
    info: Dict[str, object] = {
        "zoom": ZOOM,
        "layers": [
            "conv1 Conv2d(1, 64, 9, pad=4) + ReLU",
            "conv2 Conv2d(64, 32, 1, pad=0) + ReLU",
            "conv3 Conv2d(32, 1, 5, pad=2)",
        ],
        "params": 8129,
        "weight": weight_path(),
        "available": available(),
        "error": last_error(),
    }
    net = _cache.get("model")
    if net is not None:
        info["params"] = int(sum(p.numel() for p in net.parameters()))
        info["receptive_field"] = 17
    info["scale"] = "预放大(bicubic) + Y 通道复原"
    return info


# ---------------------------------------------------------------- inference
def _run_y(net, y: np.ndarray) -> np.ndarray:
    """对单通道 float32 [0,1] 图跑网络；分块以避免整页 64 通道特征图吃爆内存。"""
    torch = _torch
    h, w = y.shape
    out = np.empty((h, w), dtype=np.float32)
    for y0 in range(0, h, TILE):
        y1 = min(y0 + TILE, h)
        sy0, sy1 = max(0, y0 - HALO), min(h, y1 + HALO)
        for x0 in range(0, w, TILE):
            x1 = min(x0 + TILE, w)
            sx0, sx1 = max(0, x0 - HALO), min(w, x1 + HALO)
            seg = np.ascontiguousarray(y[sy0:sy1, sx0:sx1])
            t = torch.from_numpy(seg)[None, None]
            with torch.no_grad():
                o = net(t)[0, 0].numpy()
            out[y0:y1, x0:x1] = o[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0]
    return out


def _to_bgr3(bgr: np.ndarray) -> np.ndarray:
    a = np.asarray(bgr)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.ndim == 3 and a.shape[2] == 4:
        return a[:, :, :3]
    if a.ndim == 3 and a.shape[2] == 3:
        return a
    raise ValueError("upscale 需要 HxWx3 的 BGR 图，收到 %r" % (a.shape,))


def bicubic(bgr: np.ndarray, zoom: int = ZOOM) -> np.ndarray:
    """纯 bicubic 放大（与网络用的预放大完全同一路径，用于对照）。"""
    from PIL import Image

    a = _to_bgr3(np.asarray(bgr))
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).convert("YCbCr")
    pil = pil.resize((int(a.shape[1] * zoom), int(a.shape[0] * zoom)), Image.BICUBIC)
    rgb = np.asarray(pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def upscale(bgr: np.ndarray, zoom: int = ZOOM) -> np.ndarray:
    """BGR uint8 图放大 zoom 倍，返回 BGR uint8。

    严格复刻 Mirwaisse/SRCNN `run.py`：YCbCr -> 整幅 bicubic 预放大 -> 只用 Y 过网络
    -> 与 Cb/Cr 合并。只有 ``zoom=3`` 是训练尺度；其它 zoom 也做 bicubic 预放大到目标
    尺寸后同样过一遍网络（分布外，效果打折，仅作保底）。
    """
    import time

    a = _to_bgr3(np.asarray(bgr))
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    zoom = int(zoom)
    if zoom < 1:
        raise ValueError("zoom 必须 >= 1")
    if zoom == 1:
        return a.copy()

    net = _get_model()
    if net is None:
        raise RuntimeError("SRCNN 不可用：%s" % (_cache.get("err"),))

    t0 = time.perf_counter()
    from PIL import Image

    # 1) BGR -> RGB -> YCbCr，2) 整幅 bicubic 预放大（与官方一致）
    pil = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).convert("YCbCr")
    tw, th = a.shape[1] * zoom, a.shape[0] * zoom
    pil = pil.resize((tw, th), Image.BICUBIC)
    y, cb, cr = pil.split()

    # 3) 只处理 Y
    y_f = np.asarray(y, dtype=np.float32) / 255.0
    t1 = time.perf_counter()
    y_out = _run_y(net, y_f)
    t2 = time.perf_counter()

    # 4) 回到 uint8 并与 Cb/Cr 合并
    y_u8 = np.clip(y_out * 255.0, 0, 255).astype(np.uint8)
    merged = Image.merge("YCbCr", [Image.fromarray(y_u8, mode="L"), cb, cr]).convert("RGB")
    out = cv2.cvtColor(np.asarray(merged), cv2.COLOR_RGB2BGR)

    LAST.clear()
    LAST.update(
        {
            "src": (a.shape[1], a.shape[0]),
            "dst": (tw, th),
            "zoom": zoom,
            "seconds": time.perf_counter() - t0,
            "net_seconds": t2 - t1,
            "tiles": ((th + TILE - 1) // TILE) * ((tw + TILE - 1) // TILE),
            "weight": _cache.get("path"),
        }
    )
    return np.ascontiguousarray(out)


# --------------------------------------------------------------- self test
def _selftest() -> None:                     # pragma: no cover - 手工跑
    import json

    print(json.dumps(describe(), ensure_ascii=False, indent=2))
    if not available():
        raise SystemExit("SRCNN 不可用: %s" % (last_error(),))
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    out = upscale(img)
    print("random 37x53x3 ->", out.shape, out.dtype, "range", int(out.min()), int(out.max()))
    print("LAST:", LAST)


if __name__ == "__main__":                   # pragma: no cover
    _selftest()
