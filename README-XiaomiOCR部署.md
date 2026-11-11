# Xiaomi-OCR-0 本地部署（成功）

## 结论：这是目前最好的方案，且零边际成本

在 GTX 1650（4GB）上，**整页 742×1003（9px 字形）约 1 分钟**完成识别，输出同时包含：
数字、减时线、增时线 `-`、附点 `·`、反复记号 `:::`、小节线 `|`、括号、**以及逐字歌词**。

三月雨第一行的原始输出（未经任何后处理）：

```
（6i23 6i23 6i23 6i23 | 6i23 6i23 3）6i | 2·3 21 65 | 6i 23 5 63 |
细雨 坠 烟 水蒙蒙 微醺 谁人醉，春风
```

与图上内容逐字对应。这里 `i` 是"带上点的 1"（`6i23` = `6 1̇ 2 3`）。

## 部署清单

| 组件 | 位置 | 大小 |
|---|---|---|
| llama.cpp Windows CPU 版 | `E:\jianpu-env\ocr\llama` | 24 MB |
| llama.cpp **CUDA 12.4** 版 + cuBLAS 运行库 | `E:\jianpu-env\ocr\llama-cuda` | 275 + 373 MB |
| Xiaomi-OCR-0 Q4_K_M | `E:\jianpu-env\ocr\model\Xiaomi-OCR-0.Q4_K_M.gguf` | 505 MB |
| 视觉投影层 mmproj | `E:\jianpu-env\ocr\model\mmproj-bf16.gguf` | 198 MB |

驱动 560.76（GTX 1650）支持到 CUDA 12.4，所以选 12.4 而不是 13.5。

调用命令：

```powershell
$env:PATH="E:\jianpu-env\ocr\llama-cuda;$env:PATH"
E:\jianpu-env\ocr\llama-cuda\llama-mtmd-cli.exe -m ...\Xiaomi-OCR-0.Q4_K_M.gguf `
  --mmproj ...\mmproj-bf16.gguf --image page.png `
  -p "Extract the text in the image." -n 2400 --temp 0 -ngl 99
```

速度实测：视觉编码 ~50 s（一次性），文本生成 ~6 s。整页 1 分钟。

## 代码

- `jp/xiaomi_ocr.py`：调模型 + 解析成音符序列
  - `ocr_image(path)` → 原始文本
  - `parse_line(text)` → `[OcrNote(digit, octave, acc, aug_dot)]`
  - 规则：`i`→`1` 且高八度；`·`→附点（不影响音高）；`-`→增时线（跳过）；`|()`等为结构
- `jp/score_from_ocr.py`：谱行↔歌词配对 → 固定调换算 → 按歌词标点分句 → 文本谱

产物：
- `out/三月雨_ocr.txt` — OCR 原始全文
- `out/三月雨_textscore.txt` — 固定调文本谱（带歌词与拖腔）

## 待办（都不花钱）

1. **歌词行配对**：目前第 1 行（前奏）误配了第二行的歌词，需要按"谱行下方第一行汉字"更严格配对，并跳过无词的前奏行。
2. **图像输出**：OCR 给的是文本序列、没有坐标。要出"原位改字"的图，需把 OCR 序列按小节线锚定到我检测到的框上（`jp/vlmfill2.py` 里已有单调对齐 + 有墨迹才插入的逻辑，可复用）。
3. **批量化**：39 个曲目 × 1 分钟/页 ≈ 1 小时，纯本地。

## 成本

DeepSeek API **不再需要**（`jp/vlm.py` 的闸门保持默认关闭）。识别全部走本地 GPU。
