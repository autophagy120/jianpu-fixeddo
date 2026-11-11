"""画当前程序的流程图（纯 matplotlib，中文用雅黑）。输出 out/程序流程图.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

FONT = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = [FONT, "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_LOCAL = "#1d4e79"     # 本地 CPU
C_GPU = "#7a2c62"       # GPU
C_NET = "#2e6b3a"       # 联网 API
C_OUT = "#7a4a12"       # 输出/界面

fig, ax = plt.subplots(figsize=(15.5, 21), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 132)
ax.axis("off")

BOX_W = 74
X0 = 13


def box(y, title, lines, color, h=None, x=X0, w=BOX_W):
    n = len(lines)
    h = h or (3.6 + 2.55 * n)
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.6, edgecolor=color, facecolor="#f7f9fc"))
    ax.text(x + 1.6, y - 1.55, title, fontsize=15, fontweight="bold", color=color, va="top")
    ty = y - 4.6
    for ln in lines:
        ax.text(x + 2.4, ty, ln, fontsize=11.4, color="#1a1a1a", va="top")
        ty -= 2.55
    return y - h


def arrow(y1, y2, x=X0 + BOX_W / 2, label=""):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=20,
                                 linewidth=1.6, color="#5a6673"))
    if label:
        ax.text(x + 1.5, (y1 + y2) / 2, label, fontsize=10.5, color="#5a6673", va="center")


y = 130
y = box(y, "① 导入（图片 / 文件夹 / 路径）", [
    "浏览器选择器上传 or 粘贴路径递归扫描；排除 (原调…) 对照图",
    "内容哈希去重；纯红像素占比 > 6% 判为「已转调输出」，批量时跳过",
    "avif / heic 解不了的转 jpg；所有图片复制进库，不依赖原文件",
    "▸ 轮子：OpenCV(cv2.imdecode) · Pillow · hashlib · os.walk · http.server 上传",
], C_LOCAL)
arrow(y, y - 2.2)

y = box(y - 2.2, "② 本地库（自己的数据库）", [
    "library.json：每页一条记录（song/title/key/target/status/group/strength…）",
    "library/<id>/：page.jpg（原图/转码图）· ocr.txt（OCR 文本）· cache.pkl（打开缓存）",
    "缩略图 300px / 悬停预览 1100px 懒生成，垃圾箱软删除（永不复活）",
    "▸ 轮子：JSON + 文件系统 · pickle · Pillow 缩略图 · 多线程 RLock 串行写",
], C_LOCAL)
arrow(y, y - 2.2)

y = box(y - 2.2, "③ OCR 取文字（本地，零 API 费用）", [
    "llama.cpp 多模态 CLI + Xiaomi-OCR-0（Q4_K_M 505MB + mmproj 198MB，GGUF）",
    "★ 简谱专用提示词（点名要数字/八度点/减时线，否则只吐歌词）",
    "★ 兜底：RapidOCR 字符框按行拼数字行（VLM 漏行时救回整页）",
    "★ 调度：1 GPU + N CPU 抢同一队列（3 GPU 会挤爆 4G 显存反而慢 5-10 倍）",
    "▸ 轮子：llama.cpp(CUDA) · rapidocr-onnxruntime · onnxruntime · 线程队列",
], C_GPU)
arrow(y, y - 2.2)

y = box(y - 2.2, "④ AI 命名 / 自动分份（联网，只发文字）", [
    "从 OCR 文本取歌词片段 → DeepSeek Search API 搜歌名 → 结果喂模型选标题",
    "反幻觉：标题必须在搜索结果里出现过，否则留空（宁可不标）",
    "兜底：用 OCR 首行（常是曲名）再搜一次",
    "分份：按标题聚类 + 目录 + 时间间隔（μ+1.5σ）；未识别留右侧未分类",
    "▸ 轮子：DeepSeek API（/chat/completions, json_mode, thinking=disabled；/v1/tools/search）· ThreadPool 10 并发",
], C_NET)
arrow(y, y - 2.2)

y = box(y - 2.2, "⑤ 版面几何分析（每页一次，结果缓存）", [
    "Otsu 二值化 → 去掉细横线（减时线/谱线）→ 低分辨率上采样 + 锐化",
    "连通域 → 拆宽块（粘连数字）→ 合并竖笔画（4 断成两半）→ 拆粘连八度点",
    "找谱行 / 歌词行 / 系统，过滤小节线、反复号",
    "▸ 轮子：OpenCV（connectedComponentsWithStats / morphology / resize）· numpy",
], C_LOCAL)
arrow(y, y - 2.2)

y = box(y - 2.2, "⑥ 数字识别（字形分类器）", [
    "每个槽位裁 patch → 闭集分类 0-7（含「不是数字」拒绝类）",
    "▸ 轮子：自训 GlyphNet（ONNX）· onnxruntime CPU 推理 · templates.npy 模板兜底",
], C_LOCAL)
arrow(y, y - 2.2)

y = box(y - 2.2, "⑦ 文本 ↔ 版面 对齐", [
    "几何槽位按行排序 → 与 OCR 文本做全局单调 DP 对齐（不交叉、可跳字）",
    "行级策略自动选：note（全局 DP）或 measure（按小节锚定）",
    "顺带挂八度点 / 升降号；算对齐一致率（agree_rate）作为质量指标",
    "▸ 轮子：numpy 动态规划（自写 align_streams）· 无第三方对齐库",
], C_LOCAL)
arrow(y, y - 2.2)

y = box(y - 2.2, "⑧ 转调（首调 → 固定调）", [
    "音级 → 绝对音高 note_to_abs(1=? )；再 abs_to_movable 落到目标调（默认 1=C）",
    "中途转调：锚点（点选的坐标，吸附到最近一行）切段，逐音用自己的调",
    "▸ 轮子：纯 Python（12 平均律 / 等音简化），无外部库",
], C_LOCAL)
arrow(y, y - 2.2)

y = box(y - 2.2, "⑨ 渲染出图（就地替换，保留原版面）", [
    "按几何擦掉原字形（只擦非红墨）→ 画红字 + 八度点（行内统一字号/基线）",
    "没配上的墨块画橙框；转调点画紫色虚线 + 1=G 标签；未配对行不擦白（只标橙线）",
    "★ 每行统一主字号 + 统一基线，辅助字号 = 0.72×（可标例外）",
    "▸ 轮子：Pillow（ImageDraw / FreeType 字体）· numpy · 背景擦除缓存",
], C_OUT)
arrow(y, y - 2.2)

y = box(y - 2.2, "⑩ 输出 / 交互纠错", [
    "批量：out/batch/<歌>/<页>_fixed.jpg + 文本谱（首调/固定调/绝对音名/歌词）",
    "交互：本地 HTTP 服务 + 单画布（canvas），操作全自动 POST、无下载",
    "  单击改音 · 双击插入 · 拖框加强识别/清除 · 点橙框回填 · 多选批量删 ·",
    "  Ctrl+Z 回退 · 识别强度三档（严格/标准/加强）· 拖动归类 · 垃圾箱",
    "▸ 轮子：http.server(ThreadingHTTPServer) · 原生 JS/canvas · JSON 接口",
], C_OUT)

# 侧栏图例
lx, ly = 90, 128
ax.add_patch(FancyBboxPatch((lx - 1, ly - 30), 11, 30, boxstyle="round,pad=0.4",
                            linewidth=1.2, edgecolor="#9aa5b1", facecolor="#ffffff"))
ax.text(lx + 0.3, ly - 2, "图例", fontsize=13, fontweight="bold", color="#333")
items = [("本地 CPU", C_LOCAL), ("本地 GPU", C_GPU), ("联网 API", C_NET), ("输出/界面", C_OUT)]
yy = ly - 6
for name, c in items:
    ax.add_patch(FancyBboxPatch((lx + 0.6, yy - 1.2), 3.2, 2.2, boxstyle="round,pad=0.1",
                                linewidth=1.4, edgecolor=c, facecolor=c, alpha=.25))
    ax.text(lx + 4.6, yy - 0.2, name, fontsize=10.5, color="#333", va="center")
    yy -= 5.4
ax.text(lx + 0.3, yy - 2, "★ = 本轮关键改动", fontsize=10.5, color="#b3261e")

ax.text(50, 131, "简谱转调工具 · 程序流程与所用轮子", fontsize=22, fontweight="bold",
        ha="center", color="#111")
ax.text(50, 128.2, "一份谱 = 一段文字(OCR) + 一套几何(版面) 的**双流对齐**；渲染只做「就地替换」，不重排版面",
        fontsize=11.5, ha="center", color="#555")

plt.tight_layout()
out = r"D:\简谱转调项目\out\程序流程图.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
