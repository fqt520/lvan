import { removeWatermark, defaultExpand } from './core.js';
import { Engine, DetEngine, clearModelCache, hasCachedModel } from './engine.js';
import { detectWatermarks } from './detect.js';

const $ = (id) => document.getElementById(id);
const engine = new Engine();
const detEngine = new DetEngine();

const els = {
  stage: $('stage'),
  empty: $('empty'),
  viewport: $('viewport'),
  wrap: $('canvasWrap'),
  img: $('imgCanvas'),
  orig: $('origCanvas'),
  mask: $('maskCanvas'),
  preview: $('previewCanvas'),
  cursor: $('brushCursor'),
  compareTag: $('compareTag'),
  busy: $('busy'),
  busyText: $('busyText'),
  busyBar: $('busyBar'),
  busySub: $('busySub'),
  toast: $('toast'),
  brushSize: $('brushSize'),
  brushVal: $('brushVal'),
  expand: $('expand'),
  expandVal: $('expandVal'),
  modelStatus: $('modelStatus'),
};

const ctx = {
  img: els.img.getContext('2d', { willReadFrequently: true }),
  orig: els.orig.getContext('2d'),
  mask: els.mask.getContext('2d', { willReadFrequently: true }),
  preview: els.preview.getContext('2d'),
};

const MASK_COLOR = '#ff2d55';
const HISTORY_LIMIT = 12;

const state = {
  loaded: false,
  W: 0,
  H: 0,
  fileName: 'image',
  tool: 'brush',
  brush: 36,
  mode: 'auto',
  autoRun: localStorage.getItem('lvan-autorun') !== '0', // 打开图片后自动识别并去除
  scope: localStorage.getItem('lvan-scope') || 'watermark', // 自动识别范围
  format: 'png',
  expand: 0, // 0 = 自动
  view: { scale: 1, tx: 0, ty: 0 },
  history: [],
  future: [],
  maskDirty: false,
  busy: false,
};

/* ------------------------------------------------------------------ */
/* 小工具                                                              */
/* ------------------------------------------------------------------ */
let toastTimer;
function toast(msg, ms = 2600) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), ms);
}

function showBusy(text, progress = null, sub = '') {
  els.busy.hidden = false;
  els.busyText.textContent = text;
  els.busyBar.parentElement.hidden = progress === null;
  if (progress !== null) els.busyBar.style.width = `${Math.round(progress * 100)}%`;
  els.busySub.textContent = sub;
}
function hideBusy() {
  els.busy.hidden = true;
}
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

function fmtMB(b) {
  return (b / 1048576).toFixed(1) + 'MB';
}

/* ------------------------------------------------------------------ */
/* 模型                                                                */
/* ------------------------------------------------------------------ */
function onModelStatus(s) {
  const st = els.modelStatus;
  st.className = 'model-status';
  switch (s.stage) {
    case 'runtime':
      st.textContent = '正在加载推理引擎…';
      if (!els.busy.hidden) showBusy('正在加载推理引擎…', null);
      break;
    case 'cache':
      st.textContent = '从本地缓存读取模型…';
      if (!els.busy.hidden) showBusy('从本地缓存读取模型…', null);
      break;
    case 'download': {
      const p = s.total ? s.loaded / s.total : 0;
      const host = new URL(s.url, location.href).host;
      st.textContent = `下载模型 ${fmtMB(s.loaded)} / ${fmtMB(s.total)}（${host}）`;
      if (!els.busy.hidden) showBusy('首次使用，正在下载 AI 模型（仅需一次）', p, `${fmtMB(s.loaded)} / ${fmtMB(s.total)} · ${host}`);
      break;
    }
    case 'init':
      st.textContent = '正在初始化模型…';
      if (!els.busy.hidden) showBusy('正在初始化 AI 模型…', null);
      break;
    case 'ready':
      st.textContent = `✅ 模型已就绪 · ${s.backend}`;
      st.className = 'model-status ok';
      break;
  }
}

async function ensureEngine(modelBuffer) {
  if (engine.ready) return true;
  showBusy('正在准备 AI 模型…', null);
  try {
    await engine.load(onModelStatus, modelBuffer);
    return true;
  } catch (e) {
    console.error(e);
    els.modelStatus.textContent = e.message;
    els.modelStatus.className = 'model-status err';
    hideBusy();
    toast('AI 模型加载失败：' + e.message.split('\n')[0], 6000);
    return false;
  }
}

/* ------------------------------------------------------------------ */
/* 打开图片                                                            */
/* ------------------------------------------------------------------ */
function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const im = new Image();
    im.onload = () => resolve({ im, url });
    im.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('无法读取该图片（如果是 HEIC 格式，请先转换为 JPG）'));
    };
    im.src = url;
  });
}

async function openFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    if (file) toast('请选择图片文件');
    return;
  }
  try {
    const { im, url } = await loadImageFile(file);
    const W = im.naturalWidth;
    const H = im.naturalHeight;
    for (const c of [els.img, els.orig, els.mask, els.preview]) {
      c.width = W;
      c.height = H;
    }
    ctx.img.clearRect(0, 0, W, H);
    ctx.img.drawImage(im, 0, 0);
    ctx.orig.drawImage(im, 0, 0);
    URL.revokeObjectURL(url);
    Object.assign(state, { loaded: true, W, H, history: [], future: [], maskDirty: false });
    state.fileName = (file.name || 'image').replace(/\.[^.]+$/, '') || 'image';
    els.empty.hidden = true;
    els.viewport.hidden = false;
    fitView();
    updateButtons();
    updateExpandLabel();
    // 后台预加载模型
    if (!engine.ready) engine.load(onModelStatus).catch(() => {});
    if (state.autoRun) setTimeout(() => run({ auto: true }), 50);
  } catch (e) {
    toast(e.message, 4000);
  }
}

/* ------------------------------------------------------------------ */
/* 视图：缩放 / 平移                                                    */
/* ------------------------------------------------------------------ */
function applyView() {
  const { scale, tx, ty } = state.view;
  els.wrap.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  els.wrap.classList.toggle('pixelated', scale >= 3);
  updateCursorSize();
}

function fitView() {
  const r = els.viewport.getBoundingClientRect();
  const pad = 24;
  const scale = Math.min((r.width - pad * 2) / state.W, (r.height - pad * 2) / state.H, 4);
  state.view.scale = Math.max(scale, 0.02);
  state.view.tx = (r.width - state.W * state.view.scale) / 2;
  state.view.ty = (r.height - state.H * state.view.scale) / 2;
  applyView();
}

function zoomAt(factor, cx, cy) {
  const v = state.view;
  const ns = Math.min(32, Math.max(0.02, v.scale * factor));
  const k = ns / v.scale;
  v.tx = cx - (cx - v.tx) * k;
  v.ty = cy - (cy - v.ty) * k;
  v.scale = ns;
  applyView();
}

function toImage(clientX, clientY) {
  const r = els.viewport.getBoundingClientRect();
  const { scale, tx, ty } = state.view;
  return { x: (clientX - r.left - tx) / scale, y: (clientY - r.top - ty) / scale };
}

/* ------------------------------------------------------------------ */
/* 历史记录                                                            */
/* ------------------------------------------------------------------ */
function maskAlpha() {
  const d = ctx.mask.getImageData(0, 0, state.W, state.H).data;
  const a = new Uint8Array(state.W * state.H);
  for (let i = 0; i < a.length; i++) a[i] = d[i * 4 + 3] > 0 ? 1 : 0;
  return a;
}

function snapshot(includeImage) {
  return {
    mask: ctx.mask.getImageData(0, 0, state.W, state.H),
    img: includeImage ? ctx.img.getImageData(0, 0, state.W, state.H) : null,
  };
}

function pushHistory(includeImage = false) {
  state.history.push(snapshot(includeImage));
  if (state.history.length > HISTORY_LIMIT) state.history.shift();
  state.future = [];
  updateButtons();
}

function restore(entry, into) {
  into.push(snapshot(!!entry.img));
  ctx.mask.putImageData(entry.mask, 0, 0);
  if (entry.img) ctx.img.putImageData(entry.img, 0, 0);
  updateButtons();
}

function undo() {
  if (!state.history.length || state.busy) return;
  restore(state.history.pop(), state.future);
}
function redo() {
  if (!state.future.length || state.busy) return;
  restore(state.future.pop(), state.history);
}

function maskHasContent() {
  // 采样检查是否有涂抹（全量检查在大图上也只需几毫秒）
  const d = ctx.mask.getImageData(0, 0, state.W, state.H).data;
  for (let i = 3; i < d.length; i += 4) if (d[i]) return true;
  return false;
}

function updateButtons() {
  $('btnUndo').disabled = !state.history.length;
  $('btnRedo').disabled = !state.future.length;
  $('btnDownload').disabled = !state.loaded;
  $('btnRun').disabled = !state.loaded;
  $('btnDetect').disabled = !state.loaded;
}

/* ------------------------------------------------------------------ */
/* 绘制选区                                                            */
/* ------------------------------------------------------------------ */
const pointers = new Map();
let drawing = null; // {type, last, start}
let pinch = null;
let spaceDown = false;

function brushImagePx() {
  return state.brush / state.view.scale;
}

function updateCursorSize() {
  const size = state.brush;
  els.cursor.style.width = size + 'px';
  els.cursor.style.height = size + 'px';
}

function moveCursor(e) {
  const showBrush = (state.tool === 'brush' || state.tool === 'eraser') && !spaceDown && e.pointerType !== 'touch';
  els.cursor.hidden = !showBrush;
  if (!showBrush) return;
  const r = els.viewport.getBoundingClientRect();
  els.cursor.style.left = e.clientX - r.left + 'px';
  els.cursor.style.top = e.clientY - r.top + 'px';
}

function strokeTo(p) {
  const c = ctx.mask;
  c.save();
  c.globalCompositeOperation = drawing.type === 'eraser' ? 'destination-out' : 'source-over';
  c.strokeStyle = MASK_COLOR;
  c.fillStyle = MASK_COLOR;
  c.lineCap = 'round';
  c.lineJoin = 'round';
  c.lineWidth = drawing.size;
  c.beginPath();
  c.moveTo(drawing.last.x, drawing.last.y);
  c.lineTo(p.x, p.y);
  c.stroke();
  if (drawing.last.x === p.x && drawing.last.y === p.y) {
    c.beginPath();
    c.arc(p.x, p.y, drawing.size / 2, 0, Math.PI * 2);
    c.fill();
  }
  c.restore();
  drawing.last = p;
}

function drawRectPreview(a, b) {
  const c = ctx.preview;
  c.clearRect(0, 0, state.W, state.H);
  c.fillStyle = MASK_COLOR;
  c.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
  c.strokeStyle = '#fff';
  c.lineWidth = 2 / state.view.scale;
  c.setLineDash([6 / state.view.scale, 4 / state.view.scale]);
  c.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
}

function currentTool() {
  return spaceDown ? 'pan' : state.tool;
}

function onPointerDown(e) {
  if (!state.loaded || state.busy) return;
  els.viewport.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (pointers.size === 2) {
    // 双指：取消正在画的笔画，进入缩放/平移
    if (drawing && drawing.type !== 'pan') cancelDrawing();
    const [a, b] = [...pointers.values()];
    pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
    drawing = null;
    return;
  }
  if (pointers.size > 2) return;

  const tool = e.button === 1 ? 'pan' : currentTool();
  const p = toImage(e.clientX, e.clientY);
  if (tool === 'pan') {
    drawing = { type: 'pan', lastClient: { x: e.clientX, y: e.clientY } };
    els.viewport.classList.add('panning');
  } else if (tool === 'rect') {
    drawing = { type: 'rect', start: p, last: p, snap: snapshot(false) };
  } else {
    drawing = { type: tool, last: p, size: brushImagePx(), snap: snapshot(false) };
    strokeTo(p);
  }
}

function cancelDrawing() {
  if (drawing?.snap) ctx.mask.putImageData(drawing.snap.mask, 0, 0);
  ctx.preview.clearRect(0, 0, state.W, state.H);
  drawing = null;
}

function onPointerMove(e) {
  moveCursor(e);
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (pinch && pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    const cx = (a.x + b.x) / 2;
    const cy = (a.y + b.y) / 2;
    const r = els.viewport.getBoundingClientRect();
    state.view.tx += cx - pinch.cx;
    state.view.ty += cy - pinch.cy;
    zoomAt(dist / pinch.dist, cx - r.left, cy - r.top);
    pinch = { dist, cx, cy };
    return;
  }
  if (!drawing) return;
  if (drawing.type === 'pan') {
    state.view.tx += e.clientX - drawing.lastClient.x;
    state.view.ty += e.clientY - drawing.lastClient.y;
    drawing.lastClient = { x: e.clientX, y: e.clientY };
    applyView();
    return;
  }
  const p = toImage(e.clientX, e.clientY);
  if (drawing.type === 'rect') {
    drawing.last = p;
    drawRectPreview(drawing.start, p);
  } else {
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of events) strokeTo(toImage(ev.clientX, ev.clientY));
  }
}

function onPointerUp(e) {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = null;
  if (!drawing) return;
  els.viewport.classList.remove('panning');
  if (drawing.type === 'rect') {
    ctx.preview.clearRect(0, 0, state.W, state.H);
    const a = drawing.start;
    const b = drawing.last;
    if (Math.abs(a.x - b.x) > 1 && Math.abs(a.y - b.y) > 1) {
      ctx.mask.fillStyle = MASK_COLOR;
      ctx.mask.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      commitDrawing();
    }
  } else if (drawing.type !== 'pan') {
    commitDrawing();
  }
  drawing = null;
}

function commitDrawing() {
  state.history.push(drawing.snap);
  if (state.history.length > HISTORY_LIMIT) state.history.shift();
  state.future = [];
  updateButtons();
}

function onWheel(e) {
  if (!state.loaded) return;
  e.preventDefault();
  const r = els.viewport.getBoundingClientRect();
  if (e.ctrlKey || !e.deltaX) {
    const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    zoomAt(factor, e.clientX - r.left, e.clientY - r.top);
  } else {
    state.view.tx -= e.deltaX;
    state.view.ty -= e.deltaY;
    applyView();
  }
}

/* ------------------------------------------------------------------ */
/* 去水印                                                              */
/* ------------------------------------------------------------------ */
function effectiveExpand(W, H) {
  return state.expand > 0 ? state.expand : defaultExpand(W, H);
}

/** 把检测出的蒙版画到涂抹层上（红色），便于查看和修改 */
function paintMask(mask, W, H) {
  const id = ctx.mask.getImageData(0, 0, W, H);
  const d = id.data;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    d[i * 4] = 255; d[i * 4 + 1] = 45; d[i * 4 + 2] = 85; d[i * 4 + 3] = 255;
  }
  ctx.mask.putImageData(id, 0, 0);
}

async function ensureDet() {
  if (detEngine.ready) return true;
  try {
    showBusy('正在加载水印识别模型…', null);
    await detEngine.load((s) => {
      if (s.stage === 'download') showBusy('正在下载水印识别模型（约 5MB，仅需一次）', s.total ? s.loaded / s.total : 0);
    });
    return true;
  } catch (e) {
    console.error(e);
    toast('识别模型加载失败：' + e.message.split('\n')[0], 5000);
    return false;
  }
}

/** 自动识别当前图片上的水印，结果画到涂抹层。返回识别到的区域数 */
async function autoDetect() {
  if (!(await ensureDet())) return -1;
  showBusy('正在自动识别水印…', null);
  await nextFrame();
  const W = state.W, H = state.H;
  const img = ctx.img.getImageData(0, 0, W, H);
  const { mask, regions } = await detectWatermarks(img.data, W, H, detEngine.runDet, { scope: state.scope });
  if (regions.length) {
    state.history.push(snapshot(false));
    if (state.history.length > HISTORY_LIMIT) state.history.shift();
    state.future = [];
    paintMask(mask, W, H);
    updateButtons();
  }
  return regions.length;
}

async function detectOnly() {
  if (!state.loaded || state.busy) return;
  state.busy = true;
  try {
    const n = await autoDetect();
    if (n === 0) toast('没有识别到文字水印\n可以在设置里改为「所有文字」，或用画笔手动涂抹', 4000);
    else if (n > 0) toast(`识别到 ${n} 处水印（红色区域）\n可用画笔/擦除修改后点「一键去水印」`, 4000);
  } finally {
    state.busy = false;
    hideBusy();
  }
}

async function run(opts = {}) {
  if (!state.loaded || state.busy) return;
  state.busy = true;
  try {
    if (!maskHasContent()) {
      // 没有涂抹：自动识别
      const n = await autoDetect();
      if (n <= 0) {
        if (n === 0) {
          toast(opts.auto
            ? '没有自动识别到文字水印\n请用画笔涂抹水印后点「一键去水印」'
            : '没有识别到文字水印\n请用画笔涂抹（或框选）水印区域', 4500);
        }
        return;
      }
    }
    if (!(await ensureEngine())) return;
    showBusy('正在无痕去除水印…', 0);
    await nextFrame();
    const W = state.W;
    const H = state.H;
    const before = snapshot(true);
    const imgData = before.img;
    const mask = maskAlpha();
    const t0 = performance.now();
    let info = null;
    const result = await removeWatermark(imgData.data, W, H, mask, engine.runModel, {
      expand: effectiveExpand(W, H),
      mode: state.mode,
      onProgress: (d, t) => showBusy('正在无痕去除水印…', t ? d / t : 0, t > 1 ? `第 ${Math.min(d + 1, t)} / ${t} 块` : ''),
      onInfo: (i) => (info = i),
    });
    state.history.push(before);
    if (state.history.length > HISTORY_LIMIT) state.history.shift();
    state.future = [];
    ctx.img.putImageData(new ImageData(result, W, H), 0, 0);
    ctx.mask.clearRect(0, 0, W, H);
    updateButtons();
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const tr = info?.translucent?.length ? `，其中 ${info.translucent.length} 处半透明水印已还原真实像素` : '';
    toast(`✅ 完成（${secs} 秒${tr}）\n如有残留，可涂抹后再点一次；不满意可撤销`);
  } catch (e) {
    console.error(e);
    toast('处理失败：' + e.message, 5000);
  } finally {
    state.busy = false;
    hideBusy();
  }
}

/* ------------------------------------------------------------------ */
/* 保存                                                                */
/* ------------------------------------------------------------------ */
const MIME = { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp' };
const EXT = { png: 'png', jpeg: 'jpg', webp: 'webp' };

function canvasToBlob(canvas, format) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('导出失败'))), MIME[format], format === 'png' ? undefined : 0.98);
  });
}

function saveBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}

async function download() {
  if (!state.loaded) return;
  const blob = await canvasToBlob(els.img, state.format);
  saveBlob(blob, `${state.fileName}_clean.${EXT[state.format]}`);
}

/* ------------------------------------------------------------------ */
/* 批量                                                                */
/* ------------------------------------------------------------------ */
const batchResults = [];

async function runBatch(files) {
  if (!files.length) return;
  // 当前图片上有涂抹：所有图片套用同样的位置；否则每张图自动识别
  const useMask = state.loaded && maskHasContent();
  state.busy = true;
  try {
    if (!useMask && !(await ensureDet())) return;
    if (!(await ensureEngine())) return;
    $('settings').hidden = true;
    const list = $('batchList');
    list.innerHTML = '';
    batchResults.length = 0;
    $('btnBatchAll').disabled = true;
    $('batchPanel').hidden = false;
    hideBusy();

    const tmp = document.createElement('canvas');
    const tctx = tmp.getContext('2d', { willReadFrequently: true });
    const mtmp = document.createElement('canvas');
    const mctx = mtmp.getContext('2d', { willReadFrequently: true });

    for (const file of files) {
      const item = document.createElement('div');
      item.className = 'batch-item';
      item.innerHTML = `<div class="thumb"></div><div class="meta"><span class="name"></span><span class="st">等待中</span></div>`;
      item.querySelector('.name').textContent = file.name;
      list.appendChild(item);
    }
    const items = [...list.children];

    for (let k = 0; k < files.length; k++) {
      const file = files[k];
      const st = items[k].querySelector('.st');
      st.textContent = '处理中…';
      try {
        const { im, url } = await loadImageFile(file);
        const W = im.naturalWidth;
        const H = im.naturalHeight;
        tmp.width = W;
        tmp.height = H;
        tctx.drawImage(im, 0, 0);
        URL.revokeObjectURL(url);
        const imgData = tctx.getImageData(0, 0, W, H);
        let mask;
        if (useMask) {
          // 当前选区按比例缩放到这张图
          mtmp.width = W;
          mtmp.height = H;
          mctx.clearRect(0, 0, W, H);
          mctx.drawImage(els.mask, 0, 0, W, H);
          const md = mctx.getImageData(0, 0, W, H).data;
          mask = new Uint8Array(W * H);
          for (let i = 0; i < mask.length; i++) mask[i] = md[i * 4 + 3] > 0 ? 1 : 0;
        } else {
          st.textContent = '识别中…';
          await nextFrame();
          const det = await detectWatermarks(imgData.data, W, H, detEngine.runDet, { scope: state.scope });
          if (!det.regions.length) {
            st.textContent = '未识别到水印';
            items[k].querySelector('.thumb').style.backgroundImage = `url(${URL.createObjectURL(file)})`;
            continue;
          }
          mask = det.mask;
          st.textContent = '去除中…';
        }
        await nextFrame();
        const out = await removeWatermark(imgData.data, W, H, mask, engine.runModel, {
          expand: effectiveExpand(W, H),
          mode: state.mode,
        });
        tctx.putImageData(new ImageData(out, W, H), 0, 0);
        const blob = await canvasToBlob(tmp, state.format);
        const name = `${file.name.replace(/\.[^.]+$/, '')}_clean.${EXT[state.format]}`;
        batchResults.push({ blob, name });
        const href = URL.createObjectURL(blob);
        items[k].querySelector('.thumb').style.backgroundImage = `url(${href})`;
        st.innerHTML = '';
        const a = document.createElement('a');
        a.href = href;
        a.download = name;
        a.textContent = '保存';
        st.appendChild(a);
      } catch (e) {
        console.error(e);
        st.textContent = '失败';
      }
    }
    $('btnBatchAll').disabled = !batchResults.length;
  } finally {
    state.busy = false;
    hideBusy();
  }
}

/* ------------------------------------------------------------------ */
/* 设置                                                                */
/* ------------------------------------------------------------------ */
function updateExpandLabel() {
  els.expandVal.textContent = state.expand > 0
    ? `${state.expand} 像素`
    : state.loaded ? `自动（${defaultExpand(state.W, state.H)} 像素）` : '自动';
}

function bindSeg(selector, attr, key) {
  document.querySelectorAll(selector).forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll(selector).forEach((x) => x.classList.toggle('active', x === b));
      state[key] = b.dataset[attr];
    });
  });
}

function setTool(t) {
  state.tool = t;
  document.querySelectorAll('.tool').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
  els.viewport.classList.toggle('pan', t === 'pan');
}

function setBrush(v) {
  state.brush = Math.max(4, Math.min(160, Math.round(v)));
  els.brushSize.value = state.brush;
  els.brushVal.textContent = state.brush;
  updateCursorSize();
}

/* ------------------------------------------------------------------ */
/* 事件绑定                                                            */
/* ------------------------------------------------------------------ */
function init() {
  const fileInput = $('fileInput');
  const openPicker = () => fileInput.click();
  $('btnOpen').addEventListener('click', openPicker);
  $('btnOpen2').addEventListener('click', (e) => { e.stopPropagation(); openPicker(); });
  els.empty.querySelector('.drop').addEventListener('click', openPicker);
  fileInput.addEventListener('change', () => {
    openFile(fileInput.files[0]);
    fileInput.value = '';
  });

  // 拖拽 / 粘贴
  els.stage.addEventListener('dragover', (e) => { e.preventDefault(); els.stage.classList.add('dragover'); });
  els.stage.addEventListener('dragleave', () => els.stage.classList.remove('dragover'));
  els.stage.addEventListener('drop', (e) => {
    e.preventDefault();
    els.stage.classList.remove('dragover');
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) openFile(f);
  });
  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith('image/'));
    if (item) openFile(item.getAsFile());
  });

  // 画布交互
  const vp = els.viewport;
  vp.addEventListener('pointerdown', onPointerDown);
  vp.addEventListener('pointermove', onPointerMove);
  vp.addEventListener('pointerup', onPointerUp);
  vp.addEventListener('pointercancel', onPointerUp);
  vp.addEventListener('pointerleave', () => (els.cursor.hidden = true));
  vp.addEventListener('wheel', onWheel, { passive: false });
  vp.addEventListener('contextmenu', (e) => e.preventDefault());
  window.addEventListener('resize', () => state.loaded && fitView());

  // 工具
  document.querySelectorAll('.tool').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
  els.brushSize.addEventListener('input', () => setBrush(+els.brushSize.value));
  $('btnClear').addEventListener('click', () => {
    if (!state.loaded || !maskHasContent()) return;
    pushHistory(false);
    ctx.mask.clearRect(0, 0, state.W, state.H);
  });
  $('btnFit').addEventListener('click', () => state.loaded && fitView());
  $('btnRun').addEventListener('click', () => run());
  $('btnDetect').addEventListener('click', detectOnly);
  $('btnUndo').addEventListener('click', undo);
  $('btnRedo').addEventListener('click', redo);
  $('btnDownload').addEventListener('click', download);

  // 按住对比
  const cmp = $('btnCompare');
  const showOrig = (on) => {
    if (!state.loaded) return;
    els.orig.hidden = !on;
    els.mask.style.visibility = on ? 'hidden' : '';
    els.compareTag.hidden = !on;
  };
  cmp.addEventListener('pointerdown', (e) => { e.preventDefault(); showOrig(true); });
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) cmp.addEventListener(ev, () => showOrig(false));

  // 设置面板
  $('btnSettings').addEventListener('click', () => ($('settings').hidden = !$('settings').hidden));
  $('btnCloseSettings').addEventListener('click', () => ($('settings').hidden = true));
  bindSeg('[data-mode]', 'mode', 'mode');
  bindSeg('[data-format]', 'format', 'format');
  bindSeg('[data-scope]', 'scope', 'scope');
  bindSeg('[data-autorun]', 'autorun', 'autoRunStr');
  document.querySelectorAll('[data-scope]').forEach((b) => {
    b.classList.toggle('active', b.dataset.scope === state.scope);
    b.addEventListener('click', () => localStorage.setItem('lvan-scope', state.scope));
  });
  document.querySelectorAll('[data-autorun]').forEach((b) => {
    b.classList.toggle('active', (b.dataset.autorun === '1') === state.autoRun);
    b.addEventListener('click', () => {
      state.autoRun = b.dataset.autorun === '1';
      localStorage.setItem('lvan-autorun', state.autoRun ? '1' : '0');
    });
  });
  els.expand.addEventListener('input', () => {
    state.expand = +els.expand.value;
    updateExpandLabel();
  });
  $('btnLoadModel').addEventListener('click', async () => {
    if (state.busy) return;
    state.busy = true;
    try {
      if (await ensureEngine()) toast('AI 模型已就绪，之后可离线使用');
    } finally {
      state.busy = false;
      hideBusy();
    }
  });
  const modelInput = $('modelInput');
  $('btnLocalModel').addEventListener('click', () => modelInput.click());
  modelInput.addEventListener('change', async () => {
    const f = modelInput.files[0];
    modelInput.value = '';
    if (!f || state.busy) return;
    state.busy = true;
    try {
      showBusy('读取本地模型…', null);
      const buf = await f.arrayBuffer();
      if (await ensureEngine(buf)) toast('本地模型已加载并缓存');
    } finally {
      state.busy = false;
      hideBusy();
    }
  });
  $('btnClearCache').addEventListener('click', async () => {
    await clearModelCache();
    toast('已清除模型缓存');
  });

  // 批量
  const batchInput = $('batchInput');
  $('btnBatch').addEventListener('click', () => batchInput.click());
  batchInput.addEventListener('change', () => {
    const files = [...batchInput.files].filter((f) => f.type.startsWith('image/'));
    batchInput.value = '';
    runBatch(files);
  });
  $('btnCloseBatch').addEventListener('click', () => ($('batchPanel').hidden = true));
  $('btnBatchAll').addEventListener('click', async () => {
    for (const r of batchResults) {
      saveBlob(r.blob, r.name);
      await new Promise((res) => setTimeout(res, 400));
    }
  });

  // 键盘
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
    if (mod && e.key.toLowerCase() === 'o') { e.preventDefault(); openPicker(); return; }
    if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); download(); return; }
    if (mod) return;
    switch (e.key) {
      case ' ':
        if (!spaceDown) { spaceDown = true; els.viewport.classList.add('pan'); els.cursor.hidden = true; }
        e.preventDefault();
        break;
      case 'b': case 'B': setTool('brush'); break;
      case 'r': case 'R': setTool('rect'); break;
      case 'e': case 'E': setTool('eraser'); break;
      case 'h': case 'H': setTool('pan'); break;
      case '[': setBrush(state.brush / 1.2); break;
      case ']': setBrush(state.brush * 1.2); break;
      case '0': state.loaded && fitView(); break;
      case 'Enter': run(); break;
      case 'a': case 'A': detectOnly(); break;
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === ' ') {
      spaceDown = false;
      els.viewport.classList.toggle('pan', state.tool === 'pan');
    }
  });

  // 工具栏在手机上会换行，让画布区域始终贴着工具栏
  const tb = $('toolbar');
  const syncToolbar = () => {
    document.documentElement.style.setProperty('--tb-h', tb.offsetHeight + 'px');
    if (state.loaded) fitView();
  };
  if ('ResizeObserver' in window) new ResizeObserver(syncToolbar).observe(tb);
  syncToolbar();

  setBrush(state.brush);
  updateButtons();

  // 已缓存模型时静默预热（未缓存时等用户打开图片后再下载）
  hasCachedModel().then((has) => {
    if (has) engine.load(onModelStatus).catch(() => {});
  });
}

init();

// 供自动化测试使用
window.__lvan = { state, engine, openFile, run };
