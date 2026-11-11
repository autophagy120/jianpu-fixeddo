"""生成人工标注网页（自包含，双击即可用，不需要服务器）。

- 输入：models/templ_bank.npz（语料级字形聚类：308 簇 + 各簇出现次数）
- 输出：out/label_page.html（内嵌字形图，点选 0-7 或 other，一键导出 JSON）

标注完成后：页面上"导出 JSON"→ 把内容存成 out/labels.json，
然后运行 python -m jp.apply_labels out/labels.json 写回模板库。
"""
from __future__ import annotations

import base64
import json
import os
from typing import List, Sequence

import cv2
import numpy as np

from . import templbank as TB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_HTML = os.path.join(ROOT, "out", "label_page.html")
OUT_MONTAGE = os.path.join(ROOT, "out", "label_montage.png")


def png_b64(mask: np.ndarray, tile: int = 108, up: int = 3) -> str:
    """字形 mask -> 放大的白底黑字 PNG 的 base64"""
    img = 255 - (mask * 255).astype(np.uint8)
    img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_NEAREST)
    ok, enc = cv2.imencode(".png", img)
    return base64.b64encode(enc.tobytes()).decode()


def build(top_n: int = 60):
    got = TB.load()
    if not got:
        raise SystemExit("还没有模板库：先跑 python -m jp.templbank build")
    cs, labels, counts = got
    order = list(np.argsort(-counts)[:top_n])
    items = []
    for k, idx in enumerate(order):
        items.append({"k": k, "idx": int(idx), "count": int(counts[idx]),
                      "old": str(labels[idx]), "img": png_b64(cs[idx])})

    html = (_HTML_TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
            .replace("__PRE__", json.dumps(PRELABELS, ensure_ascii=False)))
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf8") as fh:
        fh.write(html)

    # 同时给一张静态拼图备用
    tile, cols = 108, 10
    rows = int(np.ceil(len(items) / cols))
    canvas = np.full((rows * (tile + 34), cols * (tile + 12), 3), 255, np.uint8)
    for it in items:
        r, c = divmod(it["k"], cols)
        y0, x0 = r * (tile + 34) + 28, c * (tile + 12) + 6
        m = cs[it["idx"]]
        big = cv2.resize(255 - m * 255, (tile, tile), interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0 + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, f"#{it['k']}", (x0 + 2, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 210), 2)
    from . import core
    core.imwrite(OUT_MONTAGE, canvas)
    print(f"标注页: {OUT_HTML}（{len(items)} 个簇）")
    print(f"备用拼图: {OUT_MONTAGE}")


PRELABELS = {
 0:"1", 1:"3", 2:"1", 3:"2", 4:"other", 5:"0", 6:"other", 7:"2", 8:"6", 9:"other",
 10:"1", 11:"5", 12:"5", 13:"1", 14:"1", 15:"6", 16:"other", 17:"7", 19:"other",
 20:"other", 21:"3", 22:"5", 23:"5", 24:"1", 25:"other", 26:"other", 27:"3", 28:"5", 29:"6",
 30:"0", 31:"other", 32:"other", 33:"other", 34:"other", 35:"other", 36:"other", 37:"3",
 38:"other", 39:"other", 40:"other", 41:"other", 42:"other", 43:"1", 44:"other", 45:"other",
 47:"0", 49:"1", 50:"other", 51:"other", 54:"6", 55:"other", 56:"other", 57:"1",
 58:"other", 59:"other",
}

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh"><meta charset="utf-8">
<title>简谱字形标定</title>
<style>
 body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:18px;background:#fafafa;color:#222}
 h1{font-size:18px;margin:0 0 6px}
 .tip{color:#666;font-size:13px;margin-bottom:12px;line-height:1.6}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:10px}
 .cell{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px;text-align:center;position:relative}
 .cell.done{border-color:#2a8;background:#f3fbf7}
 .cell img{width:108px;height:108px;image-rendering:pixelated;background:#fff}
 .no{font-size:12px;color:#888}
 .cnt{font-size:11px;color:#aaa}
 .btns{display:flex;flex-wrap:wrap;gap:3px;justify-content:center;margin-top:6px}
 .btns button{width:24px;height:24px;padding:0;font-size:13px;border:1px solid #ccc;background:#fff;
              border-radius:5px;cursor:pointer}
 .btns button:hover{background:#eef}
 .btns button.sel{background:#2a8;color:#fff;border-color:#2a8}
 .bar{position:sticky;top:0;background:#fafafa;padding:8px 0 12px;z-index:9;border-bottom:1px solid #eee}
 .bar button{padding:7px 14px;font-size:14px;margin-right:8px;cursor:pointer}
 textarea{width:100%;height:140px;font-family:ui-monospace,monospace;font-size:12px}
 #stat{font-weight:600;color:#06c}
</style>
<h1>简谱字形标定</h1>
<div class="tip">
 每一格是一个字形（放大了 3 倍）。点下面的数字给它标注；不是完整数字就点 <b>×</b>（other）。<br>
 快捷键：点选格子后按 0-7 或 x；↑↓ 跳格。<b>标完点"导出 JSON"，把内容发我。</b>
</div>
<div class="bar">
  <span id="stat">已标 0 / 0</span>
  &nbsp;&nbsp;<button onclick="exportJSON()">导出 JSON</button>
  <button onclick="copyJSON()">复制到剪贴板</button>
  <button onclick="clearAll()">清空</button>
</div>
<div class="grid" id="grid"></div>
<h3 style="margin-top:20px">导出结果（也可下载 labels.json）</h3>
<textarea id="out" placeholder="点上面的"导出 JSON""></textarea>
<script>
const ITEMS = __ITEMS__;
const PRE = __PRE__;
const LABELS = Object.assign({}, PRE);   // k -> digit|other
let cur = 0;
function render(){
  const g = document.getElementById('grid'); g.innerHTML='';
  ITEMS.forEach((it,i)=>{
    const d = document.createElement('div');
    d.className='cell'+(LABELS[it.k]!==undefined?' done':'');
    d.innerHTML = `<div class="no">#${it.k}</div>
      <img src="data:image/png;base64,${it.img}">
      <div class="cnt">出现 ${it.count} 次</div>`;
    const bs = document.createElement('div'); bs.className='btns';
    ['0','1','2','3','4','5','6','7','×'].forEach(v=>{
      const b=document.createElement('button');
      b.textContent=v;
      const val = v==='×' ? 'other' : v;
      if(LABELS[it.k]===val) b.className='sel';
      b.onclick=()=>{ LABELS[it.k]=val; cur=i; render(); };
      bs.appendChild(b);
    });
    d.appendChild(bs); g.appendChild(d);
  });
  document.getElementById('stat').textContent =
    `已标 ${Object.keys(LABELS).length} / ${ITEMS.length}`;
}
function exportJSON(){
  const m = {};
  Object.keys(LABELS).forEach(k=>{ m[ITEMS[k].idx] = LABELS[k]; });
  document.getElementById('out').value = JSON.stringify(m, null, 1);
}
function copyJSON(){
  exportJSON();
  const t=document.getElementById('out'); t.select(); document.execCommand('copy');
  alert('已复制');
}
function clearAll(){ Object.keys(LABELS).forEach(k=>delete LABELS[k]); render(); }
document.addEventListener('keydown',e=>{
  const k = e.key;
  if('01234567xX'.includes(k)){ LABELS[ITEMS[cur].k] = (k==='x'||k==='X')?'other':k; render();
    cur=Math.min(ITEMS.length-1,cur+1); e.preventDefault(); }
  else if(k==='ArrowDown'){ cur=Math.min(ITEMS.length-1,cur+1); e.preventDefault(); }
  else if(k==='ArrowUp'){ cur=Math.max(0,cur-1); e.preventDefault(); }
});
render();
</script>
</html>
"""


if __name__ == "__main__":
    build()
