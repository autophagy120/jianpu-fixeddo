# easytr（简谱转调工具）真接口逆向笔记

目标：不碰 GUI，直接调用它自己的算法，生成**未被人手改过的原始输出**用于对比。

## 结论：接口已打开（可 import、可调用）

`Jianpu Easy Trans.exe` 是 PyInstaller + Cython 打包，算法被拆成 5 个 Cython 模块
（`.pyd`，**cp36-win32**）。要用它们必须同架构解释器：

1. 下载并静默安装 **32 位 Python 3.6.8** → `work\py36_32\`（已装好，可用）
2. `python -m ensurepip` 引导 pip，然后装
   `numpy==1.19.5 scipy==1.5.4 Pillow==8.4.0`（已装好）
3. `work\et_shim.py` 注入 `scipy.misc.imresize`（scipy≥1.3 已删除，
   `getLeter`/`reco_note` 在 import 时会用到），**必须在 import 前先 `import et_shim`**

跑法：
```
D:\简谱转调项目\work\py36_32\python.exe  your_script.py
# 脚本内：
import sys; sys.path.insert(0, r'D:\简谱转调项目\work'); sys.path.insert(0, r'<exe目录>')
import et_shim
import getLeter, read_letter, reco_note, noteTranslate, replaceimgTone
```

`ModifyNotes` 依赖 PyQt5（只做人工修正），**不需要，跳过**。

## 已验证的模块与函数（参数个数已测出）

| 函数 | 参数数 | 已确认的语义 |
|---|---|---|
| `getLeter.getLetterLoc` | 2 | 内部会做 `> float` 比较，第 2 参很可能是字高中位数阈值 |
| `getLeter.removeMistaken` | 4 | 第 1 参可被下标访问（list/array） |
| `getLeter.remove_noise_getSection` | 1 | 要 numpy 数组（scipy.ndimage 报 rank 错） |
| `read_letter.read_letter` | 4 | 某参支持 `.max()` → numpy 数组 |
| `reco_note.reco_note` | 4 | 第 1 参可下标访问 |
| `reco_note.isDot` | 1 | numpy 数组 |
| `reco_note.has_sharp_b` | 3 | 某参有 `.shape` → numpy 数组 |
| `noteTranslate.noteTranslate` | 4 | 见下 |
| `replaceimgTone.replaceImgTone` | 7 | 某参 `.max()` → numpy 数组（原图） |
| `replaceimgTone.replaceImfo` | 4 | 第 1 参要能 `.load()` → **PIL Image**；第 4 参要可下标 |

## noteTranslate 的签名（已缩小到最后一层）

`noteTranslate(a, b, c, d)`：
- `a`、`b`、`d` 必须是**调名列表里的值**：`'C'` 有效；`'bA'` 在第 4 位无效
  （说明第 4 个参数用的是另一个列表，可能只含自然调或 `#X` 写法）
- `c` 必须是 **dict**（传字符串会得到 `TypeError: string indices must be integers`；
  传 list/array 会得到 `ValueError: <该对象> is not in list`）
  → 大概率就是 `replaceImfo()` 返回的"谱面信息"对象

下一步只需确认两件事：
1. `replaceImfo(Image, ?, ?, ?)` 的返回值结构（第 2/3 参传字符串会触发
   `UnicodeEncodeError: latin-1` —— 说明它往图上写字，那两参可能是**文件名/路径**；
   第 4 参要可下标，可能是 `{"原调":..,"转调":..}` 这样的 dict）
2. 该 dict 的键名 → 传给 `noteTranslate` 的第 3 参

拿到这两点后，CLI 就是：
```python
imfo = replaceImfo(pil_img, ?, ?, ?)
tab  = noteTranslate(key_from, key_to, imfo, key_to)
out  = replaceImgTone(img_arr, letters, notes, tab, key_from, key_to, ?)
```

## 已排除的路（别再试）

- 双击 exe：报 `could not find or load the Qt platform plugin "windows"`
  → 需设 `QT_QPA_PLATFORM_PLUGIN_PATH=<exe目录>\PyQt5\Qt\plugins\platforms`
- UIAutomation 枚举控件：**0 个**（PyQt 未开无障碍桥，`QT_ACCESSIBILITY=1` 也无效）
- `PostMessage(WM_LBUTTONDOWN/UP)` 点击：Qt **忽略**合成消息（点完窗口截图字节完全不变）
- 64 位 Python 3.10 import 那些 `.pyd`：架构不匹配，不可能

---

# 重大进展：签名全部拿到（从 Cython 的 co_varnames 直接读出）

Cython 模块的函数虽然没 docstring，但 `f.__code__.co_varnames` 暴露了**参数名**：

```
getLetterLoc(im_input_gray, templates)                     # 2
removeMistaken(letters, notes, box_locs, size_im)           # 4
remove_noise_getSection(bwim_input)                         # 1
read_letter(imagn, templates, thresh, type)                 # 4
reco_note(im_gray, imbw, position_region, templates)        # 4
isDot(bw_region)                                            # 1
has_sharp_b(gray_region, bw_region, templates)              # 3
noteTranslate(noteNums, notes_, toneOri, toneOut)           # 4
replaceImfo(img, str_toneOri, str_toneOut, font)            # 4
replaceImgTone(im_input_rgb, letters, notes, box_locs,
               box_notes, tone_Ori, tone_Out)               # 7
```

注意：`noteTranslate` 的第 1 参叫 `noteNums`（要能 `dict['键']` 访问）、
第 3/4 参是 `toneOri`/`toneOut`（必须是调名列表里的值，`'C'` 可用）。
`replaceImfo(img, str_toneOri, str_toneOut, font)` 第 1 参是 **PIL Image**，
第 2/3 参是调名字符串，第 4 参 `font`。

## 第一阶段已实跑通过

```python
im_gray = np.array(Image.open(p).convert('L'))
templates = np.load('templates.npy')          # (1009, 30)
ret = getLetterLoc(im_gray, templates)        # 742x1003 图：54 秒
# 返回 4 元组：
#   [0] (11,1) letters     值形如 2,2,6,2   —— 每行的字母/小节计数
#   [1] (11,4) notes       全 0（该阶段未填）
#   [2] (11,4) box_locs    形如 [258,261,200,206]  —— 两个坐标区间
#   [3] (0,4)  box_notes   空
```

已存 `work\et_stage1.pkl` 供后续解码。**性能提示**：1721x2435 的
「为了你唱下去」在这版 32 位解释器上 5 分钟没跑完（超时），
三月雨 742x1003 是 54 秒 —— 推测原程序会在识别前把图缩到某个最大边长，
这也解释了它为什么在低分辨率页上更准。下一步应先用较小尺寸喂它。

## 剩余需要确认（按顺序）

1. `box_locs` 的四个值到底是 `(x1,x2,y1,y2)` 还是 `(y1,y2,x1,x2)` —— 拿它的坐标
   在图上画框、和实际谱行对一下即可确定。
2. `reco_note(im_gray, imbw, position_region, templates)` 的 `position_region`
   应取自 `box_locs` 的一行；`imbw` 是二值图（`im_gray < 阈值`）。
3. `noteTranslate(noteNums, notes_, toneOri, toneOut)` 的 `noteNums` 是 dict，
   键名可从 `noteTranslate.pyd` 里的中文串推断：
   `音符数字 / 升号 / 降号 / 高音号 / 低音号`
4. `replaceImgTone` 的 `letters/notes/box_locs/box_notes` 就是前两步的产物。

## 关键坑（再次强调）

- 必须先 `import et_shim`（注入 `scipy.misc.imresize`）
- 必须用 `work\py36_32\python.exe`（32 位 3.6.8）
- `templates.npy` 和 `msyh.ttf` 在 exe 目录里，工作目录要在那里

---

# 完整分析结果（第二轮）

## 1. 参数文档已从二进制完整还原

`noteTranslate.pyd` @50093：
```
实现调性的转换
音符数字                       -> noteNums
升号  降号  高音号  低音号      -> notes_
原调 如 D                      -> toneOri
需要转成的调 如 C               -> toneOut
需要转成的调的图片              -> 返回值之一
```

`replaceimgTone.pyd` @60392：
```
根据识别到图像中的所有数据进行替换
所有数字 的矩阵                -> letters
数字对应的 升降号 高低音号       -> notes
数字对应的位置                  -> box_locs
升、降、高音、低音 的位置         -> box_notes
原 im_input_rgb 图中的升降高低号 -> tone_Ori / tone_Out 之外的第 5 组
```

`reco_note.pyd` @86132：
```
用于识别升降号
灰度图      -> gray_region
二值图      -> bw_region
识别到数字的区域 -> position_region
识别是否有高音号 -> （局部逻辑）
```

## 2. 各阶段实跑结果（三月雨 742x1003）

| 调用 | 耗时 | 返回 |
|---|---|---|
| `remove_noise_getSection(bw)` | 5.8s | `(清理后二值图 (1003,742), Y_section (1003,) bool)` ✅ 正常 |
| `getLetterLoc(gray, templates)` | 16~54s | 4 元组，但 `letters` 仅 **11** 条、`notes` **全零**、`box_notes` **空** |

`box_locs` 的 11 行形如 `[258,261,200,206]`，坐标约定为 **(y1,y2,x1,x2)**，
每块只有 4~6 px —— 是被当成"字母"的零星小块，不是音符。

## 3. 关键发现：原工具本身在三月雨上就失效

把同一张图缩放到不同宽度喂进去：

| 输入宽度 | letters | notes 非零 | 耗时 |
|---|---|---|---|
| **742（原尺寸）** | **11** | 0 | 16s |
| 1000 | 0 | 0 | 4s |
| 1400 | 0 | 0 | 11s |

- 只有原尺寸能检出东西，缩放后直接归零 → 这个实现对分辨率极其敏感
- 即便是原尺寸，也只检出 11 个碎块、一个音符都没认出来
  （日志里 `Mean of empty slice` 佐证内部在空数组上取中位数）

**这与用户的说法吻合**：语料里三月雨的 easytr 输出是**手动修正过**的，
说明工具自己在这一页上基本没识别出来。

## 4. 结论与判断

- 逐行裁剪喂 `getLetterLoc`（50 px 高的条带）→ 返回 0；整页才返回 11。
  说明它期望**整页**输入，内部自己做行切分。
- 内部还有 `SELECTOR` 这个局部/模块级开关没弄清，`notes` 恒零很可能与它或
  与内部二值化阈值有关；但**即使打通，原工具在低分辨率页上的基线也很低**。

因此"用 easytr 原始输出做基准来比较"这件事的价值要重新评估：
在高清页（万神纪、Bad Apple 等）它大概率正常，在三月雨这种 9px 页上它本身就不行。

## 5. 已产出的可复用资产

- `work\py36_32\`  32 位 Python 3.6.8 完整环境（numpy/scipy/Pillow 已装）
- `work\et_shim.py`  scipy.misc.imresize 兼容垫片（import 前必须加载）
- `work\et_decompile.py`  元信息提取脚本，输出 `work\et_decompile\*.txt`
- `work\et_strings.txt`  从 pyd 二进制提取的全部中文串（含参数字档）
- `work\et_stage1.pkl`  第一阶段返回数据
- `work\启动easytr.bat`  正确启动 GUI（Qt 插件路径已修）
