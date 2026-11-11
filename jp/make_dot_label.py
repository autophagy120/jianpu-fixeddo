"""生成「八度点」标注网页：一屏一个音，看清上下有几个点，按键确认/改。

为什么标点而不是标数字：
数字识别是 OCR 已经做完的部分——实测为了你唱下去 98.7%、三月雨 80%，
三条模型路线（从零训 38-64%、OrpheusNet 预训练 85%、字体模板 13-36%）
都没有超过现有的 87%。再标 360 个数字，测的是已经饱和的能力。

而用户反馈过的**每一个**错误都属于结构层，不属于数字层：
`6 1̇ 2̇ 3̇` 只认出第一个点（八度点）、歌词行中间偶尔漏一两个音（行分类）、
很清晰的一行识别不到（行分类）、行距不标准（行分割）、反复段表达不了（结构）。

所以真值要标在八度点上——它是目前唯一还在错、且错误会直接污染成品的环节。
裁剪时上下都留足余量，让原始的点完整露出来。

输出：out/dot_label.html
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

PAGES: List[Tuple[str, str]] = [
    ("为了你唱下去", r"work\为了你唱下去.jpg"),
    ("BadApple", r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\Bad Apple！！\371057593e3fc10d8bb [原始大小].jpg"),
    ("三月雨", r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg"),
]

PER_PAGE = 120
SHOW_H = 150          # 裁剪后显示高度


def _b64(gray: np.ndarray) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _crop_note(gray: np.ndarray, s: core.Note) -> str:
    """上下各留 0.62 个字高，保证八度点完整露出。

    左右留 0.65 个字高**故意留得很宽**：`4` 这类字形上下两笔在印刷体里是断开的
    两个连通块，被当成两个字，槽位框只有真字形的一半（实测 h=18 vs 中位 28）。
    标注图必须让人看到完整字形，否则是在让人认半截图。
    """
    import cv2
    h, w = gray.shape
    x0, y0, x1, y1 = s.box
    gh = max(1.0, float(y1 - y0))
    pad_y = 0.62 * gh
    pad_x = 0.65 * gh
    a = max(0, int(x0 - pad_x))
    b = max(0, int(y0 - pad_y))
    c = min(w, int(x1 + pad_x))
    d = min(h, int(y1 + pad_y))
    if c <= a or d <= b:
        return ""
    sub = gray[b:d, a:c]
    k = SHOW_H / float(max(1, sub.shape[0]))
    sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return _b64(sub)


def _crop_line(gray: np.ndarray, row_notes: List[core.Note], s: core.Note) -> str:
    """上下文：同一行的整条，缩到小图。"""
    import cv2
    h, w = gray.shape
    if not row_notes:
        return ""
    y0 = min(n.box[1] for n in row_notes)
    y1 = max(n.box[3] for n in row_notes)
    gh = max(1.0, float(y1 - y0))
    a = max(0, int(min(n.box[0] for n in row_notes) - 0.5 * gh))
    c = min(w, int(max(n.box[2] for n in row_notes) + 0.5 * gh))
    b = max(0, int(y0 - 0.62 * gh))
    d = min(h, int(y1 + 0.62 * gh))
    if c <= a or d <= b:
        return ""
    sub = gray[b:d, a:c]
    # 标出当前音的位置（红线），避免长行里找不到
    sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
    mx = int(round((s.x - a)))
    cv2.line(sub, (mx, 0), (mx, sub.shape[0] - 1), (0, 0, 220), 2)
    k = 74.0 / float(max(1, sub.shape[0]))
    sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", sub, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build(seed: int = 7) -> dict:
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
        by_row = {}
        for s in slots:
            by_row.setdefault(int(s.row), []).append(s)
        # **均匀随机抽样，不做分层**。
        # 早先按「系统判为无点/有点」分层，逻辑是错的：`pred_oct==0` 只是
        # "我的检测器没看见点"，而漏点正是要被检验的问题；拿它当"确实无点"
        # 去分层，等于用被检验的结论去挑样本，而且正好把最有价值的漏点样本
        # 挤出样本集。现在每个槽位等概率被抽。
        sel = list(slots)
        random.shuffle(sel)
        sel = sel[:PER_PAGE]
        sel.sort(key=lambda s: (int(s.row), s.x))
        for s in sel:
            row_notes = by_row.get(int(s.row), [])
            items.append(dict(
                id=nid, page=page_name,
                pred_oct=int(s.octave),        # 几何给的点数（显示出来供你确认）
                pred_dig=str(s.digit),
                img=_crop_note(page.gray, s),
                ctx=_crop_line(page.gray, row_notes, s),
            ))
            nid += 1
    return dict(meta=dict(n=len(items), per_page=PER_PAGE,
                          pages=[p[0] for p in PAGES]), items=items)


HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>八度点标注</title>
<style>
:root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--acc:#4da3ff;--ok:#31c48d;--bad:#ff7a6b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Microsoft YaHei",system-ui,sans-serif;
     display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:16px}
#bar{width:min(780px,96vw);height:6px;background:#2a2e35;border-radius:3px;overflow:hidden}
#fill{height:100%;width:0;background:var(--acc);transition:width .12s}
#hdr{width:min(780px,96vw);display:flex;justify-content:space-between;color:var(--dim);
     font-size:13px;margin:8px 0 12px}
#card{background:#fff;border-radius:10px;padding:8px;display:flex;align-items:center;justify-content:center}
#card img{display:block;max-height:160px}
#guess{margin-top:10px;font-size:15px;color:var(--dim)}
#guess b{color:#ffd166;font-size:19px}
#ctxbox{width:min(780px,96vw);margin-top:8px;display:flex;justify-content:center}
#ctx{max-width:100%;border-radius:5px;opacity:.9}
#btns{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;width:min(780px,96vw);margin-top:16px}
button{font:600 17px/1 "Microsoft YaHei",sans-serif;padding:15px 0;border:0;border-radius:9px;
       background:#2a2e35;color:var(--fg);cursor:pointer}
button:hover{background:#343a44}
button.ok{background:#1f3d2e;color:#8ee2b6}
button.up{background:#243a52;color:#a8d4ff}
button.dn{background:#3a2a3f;color:#e6b8ff}
button.no{background:#3a2a2a;color:#ffb4a8}
button.undo{background:#2f2a3a}
#hint{width:min(780px,96vw);color:var(--dim);font-size:13px;margin-top:12px;text-align:center}
kbd{background:#2a2e35;border-radius:4px;padding:1px 6px;font:12px/1.4 monospace;color:#cfd3d8}
#done{display:none;width:min(780px,96vw);text-align:center;margin-top:26px}
#done h2{color:var(--ok)}
.big{padding:15px 26px;font-size:16px;background:var(--acc);color:#04121f;margin-top:14px}
</style></head>
<body>
<div id="bar"><div id="fill"></div></div>
<div id="hdr"><span id="cnt"></span><span id="src"></span></div>
<div id="card"><img id="im" alt="note"></div>
<div id="guess"></div>
<div id="ctxbox"><img id="ctx" alt="line"></div>
<div id="btns">
  <button class="ok2" id="correct" style="grid-column:1/-1">✓ 系统判断正确　（空格）</button>
  <button class="ok"  data-v="0">没有点</button>
  <button class="up"  data-v="1">上 1 点</button>
  <button class="up"  data-v="2">上 2 点</button>
  <button class="dn"  data-v="3">下 1 点</button>
  <button class="dn"  data-v="4">下 2 点</button>
  <button class="no"  data-v="x">不是音符</button>
  <button class="no"  data-v="s">字形被切坏</button>
  <button class="undo" id="undo">撤销</button>
  <button data-v="skip" style="background:#33383f">跳过</button>
  <button id="export" style="background:#1f2b24;color:#8ee2b6">导出 JSON</button>
</div>
<div id="hint">键盘：<kbd>空格</kbd> 系统正确（最常用，一键过）· <kbd>0</kbd> 无点 · <kbd>1</kbd>/<kbd>2</kbd> 上点 1/2 · <kbd>3</kbd>/<kbd>4</kbd> 下点 1/2 · <kbd>x</kbd> 不是音符 · <kbd>s</kbd> 字形被切坏 · <kbd>p</kbd> 跳过 · <kbd>Backspace</kbd> 撤销</div>
<div id="hint" style="margin-top:6px">系统判对了就按 <kbd>空格</kbd> 一路过。判错了才按 <kbd>0</kbd>–<kbd>4</kbd> 纠正。字形残缺或低音点糊在减时线上看不清，按 <kbd>s</kbd> 或 <kbd>p</kbd> 跳过。</div>
<div id="hint" style="margin-top:6px">系统猜的点数显示在上面，<b>对了就直接按对应数字</b>，不用犹豫。</div>
<div id="done"><h2>标完了 ✓</h2><p id="dst"></p>
  <button class="big" onclick="exportJSON()">导出 JSON</button></div>

<script>
const DATA = __DATA__;
const KEY = 'jp_dot_labels_v1';
let labels = JSON.parse(localStorage.getItem(KEY) || '{}');
let stack = [], idx = 0;
const NAME = {0:'无点',1:'上1点',2:'上2点',3:'下1点',4:'下2点',x:'不是音符',s:'字形被切坏',skip:'跳过'};

function firstUnlabeled(){
  for (let i=0;i<DATA.items.length;i++) if (!(DATA.items[i].id in labels)) return i;
  return -1;
}
function save(){ localStorage.setItem(KEY, JSON.stringify(labels)); }
function mark(v){ const it=DATA.items[idx]; if(!it) return;
  labels[it.id]=v; stack.push(it.id); save(); idx=firstUnlabeled(); render(); }

// 「系统判断正确」= 直接采用系统的判断作为标签
function markCorrect(){
  const it=DATA.items[idx]; if(!it) return;
  let v;
  if(!/^[0-7]$/.test(it.pred_dig)) v='x';        // 系统压根没认成数字
  else if(it.pred_oct>0)  v=String(Math.min(2,it.pred_oct));   // 上1 -> '1', 上2 -> '2'
  else if(it.pred_oct<0)  v=String(3+Math.min(1,(-it.pred_oct)-1)); // 下1 -> '3', 下2 -> '4'
  else v='0';
  mark(v);
}
function undo(){ if(!stack.length) return; const id=stack.pop(); delete labels[id]; save();
  idx=DATA.items.findIndex(x=>x.id===id); render(); }

function accuracy(){
  const per={};
  DATA.items.forEach(it=>{
    const v = labels[it.id];
    if (v===undefined || v==='skip' || v==='x') return;
    const truth = v==='0'?0:parseInt(v,10);
    const guess = it.pred_oct>0?it.pred_oct:(it.pred_oct<0?2+(-it.pred_oct):0);
    per[it.page]=per[it.page]||[0,0]; per[it.page][1]++;
    if (truth===guess) per[it.page][0]++;
  });
  return per;
}

function render(){
  const total=DATA.items.length, done=Object.keys(labels).length;
  document.getElementById('fill').style.width=(100*done/total).toFixed(1)+'%';
  document.getElementById('cnt').textContent=done+' / '+total+' 已标';
  if (idx<0||idx>=total){
    document.getElementById('card').style.display='none';
    document.getElementById('ctxbox').style.display='none';
    document.getElementById('guess').style.display='none';
    document.getElementById('btns').style.display='none';
    document.getElementById('done').style.display='block';
    const per=accuracy();
    document.getElementById('dst').textContent=Object.entries(per)
      .map(([p,[ok,n]])=>`${p}: 已标 ${n}，八度点正确 ${ok} (${(100*ok/Math.max(1,n)).toFixed(1)}%)`).join('　');
    return;
  }
  const it=DATA.items[idx];
  document.getElementById('card').style.display='flex';
  document.getElementById('ctxbox').style.display='flex';
  document.getElementById('guess').style.display='block';
  document.getElementById('btns').style.display='grid';
  document.getElementById('done').style.display='none';
  document.getElementById('im').src=it.img;
  const c=document.getElementById('ctx');
  if(it.ctx){c.src=it.ctx;c.style.display='block';}else{c.style.display='none';}
  const g = it.pred_oct>0 ? ('上 '+it.pred_oct+' 点')
          : it.pred_oct<0 ? ('下 '+(-it.pred_oct)+' 点') : '无点';
  document.getElementById('guess').innerHTML =
    `系统判断：<b>${g}</b> &nbsp;<span style="color:#6b7280">（数字 ${it.pred_dig}）</span>`;
  document.getElementById('src').textContent=it.page+' · 第 '+(idx+1)+' 个';
}

document.querySelectorAll('button[data-v]').forEach(b=>b.addEventListener('click',()=>mark(b.dataset.v)));
document.getElementById('undo').addEventListener('click',undo);
document.getElementById('correct').addEventListener('click',markCorrect);
document.getElementById('export').addEventListener('click',exportJSON);
document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key.toLowerCase()==='z'){e.preventDefault();undo();return;}
  if(e.key==='Backspace'){e.preventDefault();undo();return;}
  if(e.key===' '){e.preventDefault();markCorrect();return;}
  const k=e.key.toLowerCase();
  if(k==='p'||k==='escape'){e.preventDefault();mark('skip');return;}
  if(['0','1','2','3','4'].includes(k)){e.preventDefault();mark(k);return;}
  if(k==='x'||k==='s'){e.preventDefault();mark(k);return;}
});
function exportJSON(){
  const rows=DATA.items.map(it=>({id:it.id,page:it.page,pred_oct:it.pred_oct,pred_dig:it.pred_dig,
    label:(it.id in labels)?labels[it.id]:null}));
  const blob=new Blob([JSON.stringify({meta:DATA.meta,labels:rows},null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='dot_labels.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),4000);
}
idx=firstUnlabeled(); render();
</script></body></html>
"""


def main(out_path: str = r"out\dot_label.html"):
    data = build()
    print("样本数:", data["meta"]["n"])
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        f.write(html)
    print("写出:", out_path, "  %.1f MB" % (os.path.getsize(out_path) / 1e6))
    return out_path


if __name__ == "__main__":
    main()
