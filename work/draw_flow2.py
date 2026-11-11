"""画当前程序的流程图（matplotlib，自动排版，中文用雅黑）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_LOCAL, C_GPU, C_NET, C_OUT = "#1d4e79", "#7a2c62", "#2e6b3a", "#7a4a12"

STAGES = [
    ("① 导入（图片 / 文件夹 / 路径）", C_LOCAL, [
        "浏览器选择器上传，或粘贴路径递归扫描；排除「(原调…)」对照图",
        "内容哈希去重；纯红像素占比 > 6% 判为「已转调输出」，批量时跳过",
        "avif / heic 解不了的转成 jpg；图片复制进库，之后不依赖原文件",
        "【轮子】OpenCV(cv2.imdecode) · Pillow · hashlib · os.walk · http.server 上传",
    ]),
    ("② 本地库（自己的数据库）", C_LOCAL, [
        "library.json：每页一条记录（song/title/key/target/status/group/strength…）",
        "library/<id>/：page.jpg（原图/转码）· ocr.txt（OCR 文本）· cache.pkl（打开缓存）",
        "缩略图 300px / 悬停预览 1100px 懒生成；垃圾箱=软删除，重扫也不会复活",
        "【轮子】JSON + 文件系统 · pickle · Pillow 缩略图 · 多线程 RLock 串行写",
    ]),
    ("③ OCR 取文字（本地，零 API 费用）", C_GPU, [
        "llama.cpp 多模态 CLI + Xiaomi-OCR-0（Q4_K_M 505MB + mmproj 198MB，GGUF）",
        "★ 简谱专用提示词（点名要数字/八度点/减时线；旧提示只吐歌词）",
        "★ 兜底：VLM 漏行时用 RapidOCR 字符框按行拼出数字行",
        "★ 调度：1 GPU + N CPU 抢同一队列（3 GPU 会挤爆 4G 显存，反而慢 5–10 倍）",
        "【轮子】llama.cpp(CUDA) · rapidocr-onnxruntime · onnxruntime · 线程队列",
    ]),
    ("④ AI 命名 / 自动分份（联网，只发文字）", C_NET, [
        "OCR 文本取歌词片段 → DeepSeek Search 搜歌名 → 结果喂模型选标题",
        "反幻觉：标题必须在搜索结果里出现过，否则留空（宁可不标）",
        "兜底：用 OCR 首行（常是曲名）再搜一次",
        "分份：按标题聚类 + 目录 + 时间间隔（μ+1.5σ）；未识别留右侧未分类",
        "【轮子】DeepSeek API：/chat/completions（json_mode, thinking=disabled）",
        "　　　　　　　　　　/v1/tools/search · ThreadPool 10 并发",
    ]),
    ("⑤ 版面几何分析（每页一次，结果缓存）", C_LOCAL, [
        "Otsu 二值化 → 去细横线（谱线/减时线）→ 低分辨率上采样 + 锐化",
        "连通域 → 拆宽块（粘连数字）→ 合并竖笔画（4 断两半）→ 拆粘连八度点",
        "找谱行 / 歌词行 / 系统；过滤小节线、反复号",
        "【轮子】OpenCV（connectedComponentsWithStats / morphology / resize）· numpy",
    ]),
    ("⑥ 数字识别（字形分类器）", C_LOCAL, [
        "每个槽位裁 patch → 闭集分类 0–7（含「不是数字」拒绝类）",
        "【轮子】自训 GlyphNet（ONNX）· onnxruntime CPU 推理 · templates.npy 模板兜底",
    ]),
    ("⑦ 文本与版面 对齐", C_LOCAL, [
        "几何槽位按行排序 → 与 OCR 文本做全局单调 DP 对齐（不交叉、可跳字）",
        "行级策略自动选：note（全局 DP）/ measure（按小节锚定）",
        "挂八度点 / 升降号；算 agree_rate 作为识别质量指标",
        "【轮子】numpy 动态规划（自写 align_streams），无第三方对齐库",
    ]),
    ("⑧ 转调（首调 → 固定调）", C_LOCAL, [
        "音级 → 绝对音高 note_to_abs(1=?)；再 abs_to_movable 落到目标调（默认 1=C）",
        "中途转调：锚点（点选坐标，吸附到最近一行）切段，逐音用自己的调",
        "【轮子】纯 Python（12 平均律 / 等音简化），无外部库",
    ]),
    ("⑨ 渲染出图（就地替换，保留原版面）", C_OUT, [
        "按几何擦掉原字形（只擦非红墨）→ 画红字 + 八度点（行内统一字号/基线）",
        "没配上的墨块画橙框；转调点画紫色虚线 + 1=G；未配对行只标橙线不擦白",
        "★ 行内统一主字号 + 统一基线；辅助字号 = 0.72×（可标例外）",
        "【轮子】Pillow（ImageDraw / FreeType 字体）· numpy · 背景擦除缓存",
    ]),
    ("⑩ 输出 / 交互纠错", C_OUT, [
        "批量：out/batch/<歌>/<页>_fixed.jpg + 文本谱（首调/固定调/绝对音名/歌词）",
        "交互：本地 HTTP + 单画布 canvas，操作全自动 POST、无下载",
        "单击改音 · 双击插入 · 拖框加强识别/清除 · 点橙框回填 · Ctrl 多选批量删",
        "Ctrl+Z 回退 · 识别强度三档（严格/标准/加强）· 拖动归类 · 垃圾箱",
        "【轮子】http.server(ThreadingHTTPServer) · 原生 JS/canvas · JSON 接口",
    ]),
]

# ---- 计算布局
BOX_W = 80
X0 = 3
GAP = 4.2
LINE_H = 3.05
PAD_TOP, PAD_BOT = 2.2, 1.6

heights = [PAD_TOP + PAD_BOT + LINE_H * len(lines) for _, _, lines in STAGES]
total = sum(heights) + GAP * (len(STAGES) - 1)
title_h = 9

fig_h = 20.0
scale = fig_h / (total + title_h)
fig, ax = plt.subplots(figsize=(13.2, fig_h), dpi=160)
W = 100
ax.set_xlim(0, W)
ax.set_ylim(0, total + title_h)
ax.axis("off")

ax.text(W / 2, total + title_h - 2.0, "简谱转调工具 · 程序流程与所用轮子",
        fontsize=21, fontweight="bold", ha="center", color="#111")
ax.text(W / 2, total + title_h - 5.4,
        "一份谱 = 一段文字(OCR) + 一套几何(版面) 的双流对齐；渲染只做「就地替换」，不重排版面",
        fontsize=10.8, ha="center", color="#555")

y = total
centers = []
for (title, color, lines), h in zip(STAGES, heights):
    ax.add_patch(FancyBboxPatch((X0, y - h), BOX_W, h,
                                boxstyle="round,pad=0.25,rounding_size=1.0",
                                linewidth=1.5, edgecolor=color, facecolor="#f8fafc"))
    ax.text(X0 + 1.8, y - 1.9, title, fontsize=13.6, fontweight="bold",
            color=color, va="top")
    ty = y - 1.9 - 2.9
    for ln in lines:
        head = ln.startswith("【轮子】") or ln.startswith("★")
        ax.text(X0 + 2.6, ty, ln, fontsize=9.7, va="top",
                color="#8a2f00" if head else "#1a1a1a",
                fontweight="bold" if head else "normal")
        ty -= LINE_H
    if y != total:
        ax.add_patch(FancyArrowPatch((X0 + BOX_W / 2, y + GAP), (X0 + BOX_W / 2, y),
                                     arrowstyle="-|>", mutation_scale=17,
                                     linewidth=1.5, color="#5a6673"))
    centers.append((y, y - h))
    y -= h + GAP

# ---- 图例
lx, ly = X0 + BOX_W + 3, total + title_h - 3
ax.add_patch(FancyBboxPatch((lx, ly - 26), 13.5, 26,
                            boxstyle="round,pad=0.3", linewidth=1.1,
                            edgecolor="#9aa5b1", facecolor="#ffffff"))
ax.text(lx + 0.9, ly - 2.4, "图例", fontsize=12, fontweight="bold", color="#333")
yy = ly - 6.6
for name, c in (("本地 CPU", C_LOCAL), ("本地 GPU", C_GPU),
                ("联网 API", C_NET), ("输出 / 界面", C_OUT)):
    ax.add_patch(FancyBboxPatch((lx + 1.0, yy - 1.4), 3.4, 2.4,
                                boxstyle="round,pad=0.05", edgecolor=c,
                                facecolor=c, alpha=.22))
    ax.text(lx + 5.2, yy - 0.2, name, fontsize=9.8, color="#333", va="center")
    yy -= 5.0
ax.text(lx + 0.9, yy - 1.6, "★ = 本轮关键改动", fontsize=9.8, color="#b3261e")

out = r"D:\简谱转调项目\out\程序流程图.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("saved", out, "content_h=", round(total + title_h, 1))
