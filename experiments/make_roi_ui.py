# -*- coding: utf-8 -*-
"""生成 ROI 手动纠错网页。

思路（用户提的）：先跑转换（红字重绘），再让人**在图上框选**未识别 / 识别错误
的区域，导出 JSON 回传。因为几何已经把每个音符的框都算出来了，所以框选可以
直接**吸附到最近的音符**，导出时带上是第几行第几个音、原来的数字和点数 ——
后续无论是人工改数、还是拿去当训练真值，都是结构化的。

产出（每首一套）：
    out/roi/<曲子>.html    自包含网页（图片用相对路径，音符框内嵌 JSON）
    out/roi/<曲子>_slots.json  该页所有音符槽位（供回传比对）

网页用法：在图上按住拖拽画框；松开后自动吸附到框内/最近的音符；
右侧列表可给每个框选打标签（未识别 / 数字错 / 高低音点错 / 擦除残留 / 其他），
可以填「正确内容」。点导出得到 roi_labels.json。
"""
from __future__ import annotations

import json
import os
from typing import List

from . import align as AL
from . import core
from . import merge_octave as MO
from . import music as M

SONGS = {
    "三月雨": (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\三月雨\50b751ebbe319 [原始大小].jpg",
             r"out\batch\_ocr\三月雨__50b751ebbe319 [原始大小].txt", "bA", "三月雨_fixed.jpg"),
    "为了你": (r"work\为了你唱下去.jpg", r"work\为了你唱下去.ocr.txt", "G", "为了你唱下去_fixed.jpg"),
    "BadApple": (r"E:\DeepSeek-Harness\工作目录\测试\转调简谱\Bad Apple！！\371057593e3fc10d8bb [原始大小].jpg",
                 r"out\batch\_ocr\Bad Apple！！__371057593e3fc10d8bb [原始大小].txt", "G", "BadApple_fixed.jpg"),
}

HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>ROI 纠错 - __SONG__</title>
<style>
body{margin:0;background:#15171b;color:#e8eaed;font:14px/1.5 "Microsoft YaHei",sans-serif}
#wrap{display:flex;height:100vh}
#left{flex:1;overflow:auto;position:relative;background:#fff}
#stage{position:relative;display:inline-block}
#stage img{display:block;user-select:none;-webkit-user-drag:none}
#cv{position:absolute;left:0;top:0}
#right{width:360px;background:#1c1f24;padding:10px;overflow:auto;border-left:2px solid #333}
h3{margin:6px 0 4px;font-size:15px;color:#9fd}
#hint{color:#8a919b;font-size:12px;margin-bottom:8px}
.roi{background:#24282e;border-radius:6px;padding:7px;margin-bottom:7px;border-left:3px solid #4da3ff}
.roi b{color:#ffd166}
select,input{background:#2c3138;color:#e8eaed;border:1px solid #3a4048;border-radius:4px;padding:3px;font-size:13px;width:100%;margin-top:3px}
button{background:#2a5f8f;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer;font-size:14px}
button:hover{background:#3573a8}
.del{background:#5a2a2a;padding:2px 8px;font-size:12px;float:right}
</style></head>
<body>
<div id="wrap">
  <div id="left"><div id="stage"><img id="im" src="__IMG__"><canvas id="cv"></canvas></div></div>
  <div id="right">
    <h3>ROI 纠错　__SONG__</h3>
    <div id="hint">在左侧图上按住拖拽画框。松开后自动吸附到最近的音符（会显示它的行号/数字/点数）。</div>
    <div id="list"></div>
    <div style="margin-top:10px"><button onclick="exportJSON()">导出 JSON</button></div>
    <div id="cnt" style="color:#8a919b;font-size:12px;margin-top:6px"></div>
  </div>
</div>
<script>
const SLOTS = __SLOTS__;
const TAGS = ["未识别","数字错","高低音点错","擦除残留","多识别","其他"];
let rois = [];      // {x,y,w,h,slot,tag,fix}
let drawing=false, sx=0, sy=0, cur=null;
const im=document.getElementById('im'), cv=document.getElementById('cv'), ctx=cv.getContext('2d');

function fit(){
  cv.width=im.clientWidth; cv.height=im.clientHeight;
  cv.style.width=im.clientWidth+'px'; cv.style.height=im.clientHeight+'px';
  redraw();
}
im.onload=fit; window.onresize=fit;

function toImg(e){
  const r=cv.getBoundingClientRect();
  const s=im.naturalWidth/im.clientWidth;
  return {x:(e.clientX-r.left)*s, y:(e.clientY-r.top)*s, s:s};
}
function nearestSlot(x,y){
  let best=null,bd=1e9;
  for(let i=0;i<SLOTS.length;i++){
    const s=SLOTS[i];
    const cx=s.x+s.w/2, cy=s.y+s.h/2;
    const d=Math.hypot(cx-x,cy-y);
    const inside=(x>=s.x-s.w&&x<=s.x+2*s.w&&y>=s.y-s.h&&y<=s.y+2*s.h);
    const sc=inside?d*0.2:d;
    if(sc<bd){bd=sc;best=i;}
  }
  return best;
}
cv.addEventListener('mousedown',e=>{const p=toImg(e);drawing=true;sx=p.x;sy=p.y;cur={x:sx,y:sy,w:0,h:0};});
window.addEventListener('mousemove',e=>{
  if(!drawing)return; const p=toImg(e);
  cur={x:Math.min(sx,p.x),y:Math.min(sy,p.y),w:Math.abs(p.x-sx),h:Math.abs(p.y-sy)};
  redraw();
});
window.addEventListener('mouseup',e=>{
  if(!drawing)return; drawing=false;
  if(cur && cur.w>4 && cur.h>4){
    const si=nearestSlot(cur.x+cur.w/2, cur.y+cur.h/2);
    rois.push({x:cur.x,y:cur.y,w:cur.w,h:cur.h,slot:si,tag:TAGS[0],fix:""});
    renderList();
  }
  cur=null; redraw();
});
function redraw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  const k=cv.width/im.naturalWidth;
  // 已配对音符的框（淡蓝），帮助定位
  ctx.strokeStyle='rgba(80,150,255,0.28)'; ctx.lineWidth=1;
  for(const s of SLOTS){ if(!s.paired) continue; ctx.strokeRect(s.x*k,s.y*k,s.w*k,s.h*k); }
  // 未配对（橙）
  ctx.strokeStyle='rgba(235,140,0,0.75)';
  for(const s of SLOTS){ if(s.paired) continue; ctx.strokeRect(s.x*k,s.y*k,s.w*k,s.h*k); }
  // 已画 ROI
  rois.forEach((r,i)=>{
    ctx.strokeStyle='#ff4d4d'; ctx.lineWidth=2;
    ctx.strokeRect(r.x*k,r.y*k,r.w*k,r.h*k);
    ctx.fillStyle='#ff4d4d'; ctx.font='14px sans-serif';
    ctx.fillText(String(i+1), r.x*k+3, r.y*k+15);
  });
  if(cur){ ctx.strokeStyle='#4dff9f'; ctx.lineWidth=2; ctx.strokeRect(cur.x*k,cur.y*k,cur.w*k,cur.h*k); }
}
function renderList(){
  const box=document.getElementById('list'); box.innerHTML='';
  rois.forEach((r,i)=>{
    const s=(r.slot!=null)?SLOTS[r.slot]:null;
    const d=document.createElement('div'); d.className='roi';
    const info=s?`行${s.row} 第${s.idx}个 数字「${s.digit}」点数${s.oct}`:'（未吸附）';
    d.innerHTML=`<button class="del" onclick="delRoi(${i})">删</button>
      <b>#${i+1}</b> ${info}
      <select onchange="setTag(${i},this.value)">${TAGS.map(t=>`<option ${t===r.tag?'selected':''}>${t}</option>`).join('')}</select>
      <input placeholder="正确内容（如 6 或 6低两点）" value="${r.fix||''}" oninput="setFix(${i},this.value)">`;
    box.appendChild(d);
  });
  document.getElementById('cnt').textContent = `已标 ${rois.length} 个 ROI`;
}
function delRoi(i){rois.splice(i,1);renderList();redraw();}
function setTag(i,v){rois[i].tag=v;}
function setFix(i,v){rois[i].fix=v;}
function exportJSON(){
  const rows=rois.map((r,i)=>{
    const s=(r.slot!=null)?SLOTS[r.slot]:null;
    return {n:i+1,roi:[Math.round(r.x),Math.round(r.y),Math.round(r.w),Math.round(r.h)],
            tag:r.tag,fix:r.fix,
            slot:(s?{row:s.row,idx:s.idx,digit:s.digit,oct:s.oct,paired:s.paired,
                     box:[Math.round(s.x),Math.round(s.y),Math.round(s.w),Math.round(s.h)]}:null)};
  });
  const blob=new Blob([JSON.stringify({song:"__SONG__",rois:rows},null,1)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='roi_labels___SONG__.json'; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),4000);
}
</script></body></html>
"""


def build(song: str) -> str:
    from .glyphnet import GlyphNet
    orig, ocr_path, key_name, img_name = SONGS[song]
    txt = MO.sanitize_ocr(open(ocr_path, encoding="utf8").read())
    lines, pairs, st, page = MO.merge_page(orig, txt, GlyphNet())
    paired_ids = {id(s) for s, _o in pairs}
    note_of = {id(s): o for s, o in pairs}

    # 槽位（工作空间坐标 = 渲染图坐标）
    slots = getattr(page, "_slots", None) or AL.ordered_slots(page, GlyphNet())
    rows: dict = {}
    out_slots: List[dict] = []
    for s in slots:
        r = int(s.row)
        rows[r] = rows.get(r, 0) + 1
        o = note_of.get(id(s))
        out_slots.append(dict(
            x=float(s.box[0]), y=float(s.box[1]),
            w=float(s.box[2] - s.box[0]), h=float(s.box[3] - s.box[1]),
            row=r, idx=rows[r],
            digit=(o.digit if o else (s.digit or "?")),
            oct=int(o.octave) if o else int(s.octave),
            paired=id(s) in paired_ids,
        ))
    d = os.path.join("out", "roi")
    os.makedirs(d, exist_ok=True)
    jpath = os.path.join(d, f"{song}_slots.json")
    json.dump(out_slots, open(jpath, "w", encoding="utf8"), ensure_ascii=False)
    html = (HTML.replace("__SONG__", song)
                .replace("__IMG__", "../" + img_name)
                .replace("__SLOTS__", json.dumps(out_slots, ensure_ascii=False)))
    hpath = os.path.join(d, f"{song}.html")
    open(hpath, "w", encoding="utf8").write(html)
    n_un = sum(1 for s in out_slots if not s["paired"])
    print(f"{song}: 槽位 {len(out_slots)}（未配对 {n_un}） -> {hpath}", flush=True)
    return hpath


def main():
    return [build(s) for s in SONGS]


if __name__ == "__main__":
    main()
