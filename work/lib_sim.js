// 用最小 DOM 桩跑一遍库页面脚本，抓初始化时的运行时错误
const fs = require('fs');
const path = 'D:\\简谱转调项目\\work\\lib_page.html';
const html = fs.readFileSync(path, 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.log('没找到 script'); process.exit(1); }
const code = m[1];

function stubEl(id) {
  const el = {
    id, _html: '', _text: '', value: '', disabled: false, checked: false,
    style: new Proxy({}, { get: () => '', set: () => true }),
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    dataset: {}, children: [], files: [], options: [],
    appendChild(c) { this.children.push(c); return c; },
    removeChild() {}, remove() {}, insertBefore(c) { return c; },
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    addEventListener() {}, removeEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
    focus() {}, click() {}, scrollIntoView() {},
    getContext() { return {}; },
    set innerHTML(v) { this._html = String(v); }, get innerHTML() { return this._html; },
    set textContent(v) { this._text = String(v); }, get textContent() { return this._text; },
  };
  return el;
}

const cache = {};
const documentStub = {
  readyState: 'complete',
  getElementById(id) { return (cache[id] = cache[id] || stubEl(id)); },
  createElement(tag) { return stubEl('_' + tag); },
  createTextNode(t) { return { t }; },
  querySelectorAll() { return []; },
  querySelector() { return null; },
  addEventListener() {},
  body: stubEl('body'),
  documentElement: stubEl('html'),
};

const libJson = {
  stats: { total: 2, new: 0, ocr: 2, done: 0, error: 0 },
  items: [
    { id: 'a1', song: '纯蓝', title: '纯蓝', status: 'done', group: 'g1', ocr_txt: 'x' },
    { id: 'a2', song: '越冰船', title: '', status: 'ocr', group: '', ocr_txt: 'y' },
  ],
  groups: { g1: { id: 'g1', name: '纯蓝', collapsed: true } },
  trash_n: 0, llm: true, ocr: true,
};
const sourcesJson = { sources: [] };

const fetchStub = (url) => Promise.resolve({
  ok: true, status: 200,
  json: () => Promise.resolve(String(url).includes('/api/lib') ? libJson : sourcesJson),
  text: () => Promise.resolve(''),
});

const errors = [];
const sandbox = {
  document: documentStub,
  fetch: fetchStub,
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  console, setTimeout, clearTimeout, setInterval: () => 0, clearInterval() {},
  Promise, JSON, Math, Date, String, Number, Array, Object, RegExp, Error,
  navigator: { userAgent: 'node' },
  location: { href: '', reload() {}, search: '' },
  alert() {}, confirm: () => true, prompt: () => '',
  Image: function () { return stubEl('img'); },
  FormData: function () { return { append() {} }; },
  URLSearchParams,
  requestAnimationFrame: (f) => setTimeout(f, 0),
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.window.addEventListener = function (ev, fn) {
  if (ev === 'error') sandbox._onerr = fn;
};
sandbox.window.onerror = null;

const vm = require('vm');
try {
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { filename: 'lib_page.js' });
} catch (e) {
  errors.push('同步异常: ' + (e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e));
}

setTimeout(() => {
  const stats = cache['stats'];
  const groups = cache['groups'];
  console.log('抓到的错误:', errors.length ? errors : '无');
  console.log('stats.innerHTML =', JSON.stringify((stats && stats.innerHTML || '').slice(0, 120)));
  console.log('groups.innerHTML 前 80 字 =', JSON.stringify((groups && groups.innerHTML || '').slice(0, 80)));
}, 400);
