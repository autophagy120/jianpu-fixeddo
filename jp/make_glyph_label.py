"""生成字形标注网页：一屏一个字形，按 0-7 点选，导出 JSON。

为什么需要它：实测三条路（从零训 CNN 38-64%、OrpheusNet 预训练 85%、字体模板
13-36%）都没有超过现有分类器的 87%。而三月雨上两个完全不同的分类器**同时卡在
75%**，说明瓶颈是**标签本身错了**——我的自动标签来自「小节内按序 zip 对 OCR」，
低分辨率页 OCR 本身就错，标签跟着错，再好的模型也只能学到错误答案。

所以先要真值。你点 300-400 个，就能量出真实准确率，并分清三月雨到底是
「OCR 读错」还是「几何切错」——两者修法完全不同。

输出：out/glyph_label.html（自包含，图片 base64 内嵌，可离线双击打开）
"""
from __future__ import annotations

import base64
import json
import os
import random
from typing import List, Tuple

import numpy as np

from . import align as AL
from . import core
from . import measures as ME

PAGES: List[Tuple[str, str]] = [
    ("为了你唱下去", r"work\为了你唱下去.jpg"),
    ("BadApple", r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\Bad Apple！！\371057593e3fc10d8bb [原始大小].jpg"),
    ("三月雨", r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"),
]

PER_PAGE = 120
GLYPH_H = 170          # 放大后的字形显示高度（像素）


def _b64(bgr_or_gray: np.ndarray, ext: str = ".png") -> str:
    import cv2
    if ext == ".jpg":
        ok, buf = cv2.imencode(ext, bgr_or_gray, [cv2.IMWRITE_JPEG_QUALITY, 72])
    else:
        ok, buf = cv2.imencode(ext, bgr_or_gray)
    if not ok:
        return ""
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return "data:%s;base64,%s" % (mime, base64.b64encode(buf.tobytes()).decode("ascii"))


def _crop(gray: np.ndarray, x0, y0, x1, y1, pad: float, target_h: int,
          ext: str = ".png") -> str:
    import cv2
    h, w = gray.shape
    ph = int(round((y1 - y0) * pad))
    a, b = max(0, int(x0) - ph), max(0, int(y0) - ph)
    c, d = min(w, int(x1) + ph), min(h, int(y1) + ph)
    if c <= a or d <= b:
        return ""
    sub = gray[b:d, a:c]
    k = target_h / float(max(1, sub.shape[0]))
    sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return _b64(sub, ext)


def build(seed: int = 42) -> dict:
    from .glyphnet import GlyphNet
    net = GlyphNet()
    random.seed(seed)
    items = []
    nid = 0
    for page_name, path in PAGES:
        if not os.path.exists(path):
            print("缺文件，跳过:", path)
            continue
        page = core.analyse(core.load_gray(path), net, path=path)
        slots = AL.ordered_slots(page, net)
        if not slots:
            continue
        med_h = float(np.median([s.box[3] - s.box[1] for s in slots]))
        # 每个字形属于哪个小节（用来给上下文小图）
        mgeo = ME.geo_measures(page, slots)
        meas_of = {}
        for m in mgeo:
            for s in m:
                meas_of[id(s)] = m
        pool = list(slots)
        random.shuffle(pool)
        pool = pool[:PER_PAGE]
        pool.sort(key=lambda s: (int(s.row), s.x))
        for s in pool:
            x0, y0, x1, y1 = s.box
            img = _crop(page.gray, x0, y0, x1, y1, 0.28, GLYPH_H)
            if not img:
                continue
            # 上下文：本小节整条（缩到同一高度）
            ctx = ""
            m = meas_of.get(id(s))
            if m:
                ctx = _crop(page.gray,
                            min(t.box[0] for t in m) - 0.2 * med_h,
                            y0,
                            max(t.box[2] for t in m) + 0.2 * med_h,
                            y1, 0.55, int(GLYPH_H * 0.55), ext=".jpg")
            items.append(dict(
                id=nid, page=page_name, x=int(s.x), y=int(y0),
                w=int(x1 - x0), h=int(y1 - y0),
                pred=str(s.digit),                  # 系统猜的（不显示给用户，避免暗示）
                pred_oct=int(s.octave),             # 几何给的八度（含在字形图里可见）
                img=img, ctx=ctx,
            ))
            nid += 1
    meta = dict(n=len(items), per_page=PER_PAGE,
                pages=[p[0] for p in PAGES])
    return dict(meta=meta, items=items)


HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>简谱字形标注</title>
<style>
:root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--acc:#4da3ff;--ok:#31c48d;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Microsoft YaHei",system-ui,sans-serif;
     display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:18px}
#bar{width:min(760px,96vw);height:6px;background:#2a2e35;border-radius:3px;overflow:hidden;margin-bottom:10px}
#fill{height:100%;width:0;background:var(--acc);transition:width .15s}
#hdr{width:min(760px,96vw);display:flex;justify-content:space-between;color:var(--dim);font-size:13px;margin-bottom:14px}
#card{background:#fff;border-radius:10px;padding:10px;display:flex;align-items:center;justify-content:center;
      min-height:200px;min-width:200px}
#card img{display:block;max-height:190px;image-rendering:auto}
#ctxbox{width:min(760px,96vw);margin-top:10px;display:flex;justify-content:center}
#ctx{max-width:100%;max-height:110px;background:#fff;border-radius:6px;padding:4px;opacity:.85}
#btns{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;width:min(760px,96vw);margin-top:20px}
button{font:600 20px/1 "Microsoft YaHei",sans-serif;padding:18px 0;border:0;border-radius:10px;
       background:#2a2e35;color:var(--fg);cursor:pointer;transition:transform .05s,background .12s}
button:hover{background:#343a44}
button:active{transform:scale(.96)}
button.dig{background:#26303d}
button.dig:hover{background:#33465c}
button.skip{background:#3a2a2a;color:#ffb4a8}
button.undo{background:#2f2a3a}
button.cb{background:#1f2b24;color:#8ee2b6;font-size:15px;padding:14px 0}
#hint{width:min(760px,96vw);color:var(--dim);font-size:13px;margin-top:14px;text-align:center}
kbd{background:#2a2e35;border-radius:4px;padding:1px 6px;font:12px/1.4 monospace;color:#cfd3d8}
#done{display:none;width:min(760px,96vw);text-align:center;margin-top:30px}
#done h2{color:var(--ok)}
.big{padding:16px 28px;font-size:17px;background:var(--acc);color:#04121f;margin-top:16px}
</style></head>
<body>
<div id="bar"><div id="fill"></div></div>
<div id="hdr"><span id="cnt"></span><span id="src"></span></div>
<div id="card"><img id="im" alt="glyph"></div>
<div id="ctxbox"><img id="ctx" alt="context"></div>
<div id="btns">
  <button class="dig" data-v="0">0</button>
  <button class="dig" data-v="1">1</button>
  <button class="dig" data-v="2">2</button>
  <button class="dig" data-v="3">3</button>
  <button class="dig" data-v="4">4</button>
  <button class="dig" data-v="5">5</button>
  <button class="dig" data-v="6">6</button>
  <button class="dig" data-v="7">7</button>
  <button class="skip" data-v="skip">跳过</button>
  <button class="undo" id="undo">撤销</button>
</div>
<div id="hint">键盘：<kbd>0</kbd>–<kbd>7</kbd> 直接标 · <kbd>空格</kbd> 跳过 · <kbd>Backspace</kbd> 撤销 · <kbd>Ctrl+Z</kbd> 撤销</div>
<div id="hint" style="margin-top:6px">看不出来（糊、粘连、不是数字）就按空格跳过 —— <b>跳过比猜错有用</b>。</div>
<div id="done">
  <h2>标完了 ✓</h2>
  <p id="dst"></p>
  <button class="big" onclick="exportJSON()">导出 JSON</button>
</div>

<script>
const DATA = __DATA__;
const KEY = 'jp_glyph_labels_v1';
let labels = JSON.parse(localStorage.getItem(KEY) || '{}');
let stack = [];
let idx = 0;

function firstUnlabeled(){
  for (let i=0;i<DATA.items.length;i++) if (!(DATA.items[i].id in labels)) return i;
  return -1;
}
function save(){ localStorage.setItem(KEY, JSON.stringify(labels)); }

function render(){
  const total = DATA.items.length;
  const done = Object.keys(labels).length;
  document.getElementById('fill').style.width = (100*done/total).toFixed(1)+'%';
  if (idx < 0 || idx >= total){
    document.getElementById('card').style.display='none';
    document.getElementById('ctxbox').style.display='none';
    document.getElementById('btns').style.display='none';
    document.getElementById('done').style.display='block';
    document.getElementById('cnt').textContent = done+' / '+total;
    document.getElementById('src').textContent = '';
    const per={};
    DATA.items.forEach(it=>{ if(it.id in labels){ per[it.page]=per[it.page]||[0,0]; per[it.page][1]++; if(labels[it.id]===it.pred) per[it.page][0]++; }});
    document.getElementById('dst').textContent =
      Object.entries(per).map(([p,[ok,n]])=>`${p}: 已标 ${n}，系统猜对 ${ok} (${(100*ok/Math.max(1,n)).toFixed(1)}%)`).join('　');
    return;
  }
  const it = DATA.items[idx];
  document.getElementById('card').style.display='flex';
  document.getElementById('ctxbox').style.display='flex';
  document.getElementById('btns').style.display='grid';
  document.getElementById('done').style.display='none';
  const im = document.getElementById('im');
  im.src = it.img;
  const ctx = document.getElementById('ctx');
  if (it.ctx){ ctx.src = it.ctx; ctx.style.display='block'; } else { ctx.style.display='none'; }
  document.getElementById('cnt').textContent = `${done} / ${total} 已标`;
  document.getElementById('src').textContent = `${it.page} · 第 ${idx+1} 个`;
}

function mark(v){
  const it = DATA.items[idx];
  if (!it) return;
  labels[it.id] = v;
  stack.push(it.id);
  save();
  idx = firstUnlabeled();
  render();
}
function undo(){
  if (!stack.length) return;
  const id = stack.pop();
  delete labels[id]; save();
  idx = DATA.items.findIndex(x=>x.id===id);
  render();
}
document.querySelectorAll('button[data-v]').forEach(b=>{
  b.addEventListener('click', ()=>mark(b.dataset.v));
});
document.getElementById('undo').addEventListener('click', undo);

document.addEventListener('keydown', e=>{
  if (e.ctrlKey && e.key.toLowerCase()==='z'){ e.preventDefault(); undo(); return; }
  if (e.key==='Backspace'){ e.preventDefault(); undo(); return; }
  if (e.key===' '){ e.preventDefault(); mark('skip'); return; }
  if (/^[0-7]$/.test(e.key)){ e.preventDefault(); mark(e.key); return; }
});

function exportJSON(){
  const rows = DATA.items.map(it=>({
    id: it.id, page: it.page, x: it.x, y: it.y, w: it.w, h: it.h,
    pred: it.pred, pred_oct: it.pred_oct,
    label: (it.id in labels) ? labels[it.id] : null
  }));
  const blob = new Blob([JSON.stringify({meta: DATA.meta, labels: rows}, null, 1)],
                        {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'glyph_labels.json';
  a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 4000);
}

idx = firstUnlabeled();
if (idx < 0) idx = -1;
render();
</script></body></html>
"""


def main(out_path: str = r"out\glyph_label.html"):
    data = build()
    print("样本数:", data["meta"]["n"], " 每页:", data["meta"]["per_page"])
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        f.write(html)
    print("写出:", out_path, "  %.1f MB" % (os.path.getsize(out_path) / 1e6))
    return out_path


if __name__ == "__main__":
    main()
