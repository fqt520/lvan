/**
 * 自动识别水印
 *
 * 1. 用 PP-OCRv4 文字检测模型（DBNet，4.7MB）找出图中所有文字区域；
 * 2. 逐个判断是不是水印：
 *    - 有“半透明叠加”特征（α 直方图在某个不透明度处有尖峰）的文字 → 水印；
 *    - 位于图片四周边缘带的文字（抖音号、小红书号、@用户名 通常在角落/底部）→ 水印；
 *    - “所有文字”模式下，所有检测到的文字都去除；
 * 3. 半透明水印沿笔画特征向外生长，把 OCR 漏检的字母（倾斜、很淡的部分）也补进来。
 *
 * 输出一张蒙版，交给 core.removeWatermark 去除。
 */
import { morphBackground, alphaHistogram, prominenceAt } from './translucent.js';

export const DET_LIMIT = 1280; // 检测时图片长边上限
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const EDGE_ZONE = 0.2; // 四周 20% 视为“水印常见位置”

/** 面积平均缩放并归一化为模型输入 (CHW) */
function prepare(rgba, W, H, w, h) {
  const x = new Float32Array(3 * w * h);
  const sx = W / w, sy = H / h;
  for (let ty = 0; ty < h; ty++) {
    const y0 = Math.floor(ty * sy), y1 = Math.max(y0 + 1, Math.min(H, Math.floor((ty + 1) * sy)));
    for (let tx = 0; tx < w; tx++) {
      const x0 = Math.floor(tx * sx), x1 = Math.max(x0 + 1, Math.min(W, Math.floor((tx + 1) * sx)));
      let r = 0, g = 0, b = 0, n = 0;
      for (let y = y0; y < y1; y++) {
        for (let xx = x0; xx < x1; xx++) {
          const p = (y * W + xx) * 4;
          r += rgba[p]; g += rgba[p + 1]; b += rgba[p + 2]; n++;
        }
      }
      const o = ty * w + tx;
      x[o] = (r / n / 255 - MEAN[0]) / STD[0];
      x[w * h + o] = (g / n / 255 - MEAN[1]) / STD[1];
      x[2 * w * h + o] = (b / n / 255 - MEAN[2]) / STD[2];
    }
  }
  return x;
}

/**
 * 文字检测
 * @param runDet (input Float32Array, w, h) => Promise<Float32Array 概率图 h*w>
 * @returns 文字框（原图坐标）
 */
export async function detectText(rgba, W, H, runDet) {
  const sc = Math.min(1, DET_LIMIT / Math.max(W, H));
  const w = Math.max(32, Math.round((W * sc) / 32) * 32);
  const h = Math.max(32, Math.round((H * sc) / 32) * 32);
  const prob = await runDet(prepare(rgba, W, H, w, h), w, h);
  const sx = W / w, sy = H / h;

  const seen = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  const boxes = [];
  for (let i = 0; i < w * h; i++) {
    if (seen[i] || prob[i] <= 0.3) continue;
    let sp = 0, count = 0, sum = 0;
    let bx0 = w, by0 = h, bx1 = 0, by1 = 0;
    stack[sp++] = i;
    seen[i] = 1;
    while (sp) {
      const p = stack[--sp];
      const x = p % w, y = (p - x) / w;
      count++;
      sum += prob[p];
      if (x < bx0) bx0 = x;
      if (x > bx1) bx1 = x;
      if (y < by0) by0 = y;
      if (y > by1) by1 = y;
      if (x > 0 && !seen[p - 1] && prob[p - 1] > 0.3) { seen[p - 1] = 1; stack[sp++] = p - 1; }
      if (x < w - 1 && !seen[p + 1] && prob[p + 1] > 0.3) { seen[p + 1] = 1; stack[sp++] = p + 1; }
      if (y > 0 && !seen[p - w] && prob[p - w] > 0.3) { seen[p - w] = 1; stack[sp++] = p - w; }
      if (y < h - 1 && !seen[p + w] && prob[p + w] > 0.3) { seen[p + w] = 1; stack[sp++] = p + w; }
    }
    const score = sum / count;
    if (count < 8 || score < 0.5) continue;
    const bw = bx1 - bx0 + 1, bh = by1 - by0 + 1;
    // DB 输出的是“收缩后”的文字核心，需要按面积/周长向外扩张（unclip）
    const D = (count * 1.6) / (2 * (bw + bh));
    const thick = count / Math.hypot(bw, bh); // 文字核心的粗细（对倾斜文字也适用）
    const textH = (thick + 2 * D) * Math.sqrt(sx * sy);
    boxes.push({
      x0: Math.max(0, Math.floor((bx0 - D) * sx)),
      y0: Math.max(0, Math.floor((by0 - D) * sy)),
      x1: Math.min(W - 1, Math.ceil((bx1 + 1 + D) * sx)),
      y1: Math.min(H - 1, Math.ceil((by1 + 1 + D) * sy)),
      score,
      textH: Math.max(4, textH),
    });
  }
  return boxes;
}

/** 判断文字框内是否有半透明叠加特征，返回 {A, C, prom} 或 null */
function translucentEvidence(rgba, W, H, box) {
  const idx = [];
  for (let y = box.y0; y <= box.y1; y++) for (let x = box.x0; x <= box.x1; x++) idx.push(y * W + x);
  if (idx.length < 50) return null;
  const r = Math.max(2, Math.min(14, Math.round(box.textH * 0.14)));
  const per = [];
  for (const C of [[255, 255, 255], [0, 0, 0]]) {
    const bg = morphBackground(rgba, W, H, box, idx, r, C[0] === 255);
    const hist = alphaHistogram(rgba, 4, bg, idx, C);
    let best = null;
    for (let k = 15; k <= 92; k++) {
      const prom = prominenceAt(hist, k);
      if (!best || prom > best.prom) best = { A: k / 100, C, prom, r };
    }
    per.push(best);
  }
  per.sort((a, b) => b.prom - a.prom);
  const best = per[0];
  if (!best || best.prom < Math.max(12, idx.length * 0.015)) return null;
  // 白字黑影（或黑字白边）两种颜色都有明显特征：带描边/投影的文字，不做笔画生长
  best.ambiguous = per[1].prom > best.prom * 0.4;
  return best;
}

/** 半透明水印沿笔画向外生长，补上 OCR 漏检的部分 */
function growTranslucent(rgba, W, H, mask, box, ev) {
  const reach = Math.max(4, Math.round(box.textH * 0.6));
  const pad = Math.round(box.textH * 4); // 最多向外长四个字高
  const region = {
    x0: Math.max(0, box.x0 - pad), y0: Math.max(0, box.y0 - pad),
    x1: Math.min(W - 1, box.x1 + pad), y1: Math.min(H - 1, box.y1 + pad),
  };
  const rw = region.x1 - region.x0 + 1, rh = region.y1 - region.y0 + 1;
  const idx = [];
  for (let y = region.y0; y <= region.y1; y++) for (let x = region.x0; x <= region.x1; x++) idx.push(y * W + x);
  const bg = morphBackground(rgba, W, H, region, idx, ev.r, ev.C[0] === 255);
  const { A, C } = ev;
  // 候选笔画像素：α 接近 A，且颜色偏移方向符合“朝水印颜色混合”
  const cand = new Uint8Array(rw * rh);
  for (let k = 0; k < idx.length; k++) {
    const p = idx[k] * 4;
    let num = 0, den = 0, d2 = 0;
    for (let c = 0; c < 3; c++) {
      const b = bg[k * 3 + c];
      const d = rgba[p + c] - b;
      num += d * (C[c] - b);
      den += (C[c] - b) * (C[c] - b);
      d2 += d * d;
    }
    if (den < 900) continue;
    const a = num / den;
    const perp2 = d2 - (num * num) / den;
    if (a > A * 0.65 && a < A * 1.4 && perp2 < 400 + 0.1 * d2) cand[k] = 1;
  }
  // 去掉细碎的候选（自然图像里的边缘只有 1~2 像素宽，而水印笔画有一定粗细）
  const strokeW = Math.max(2, box.textH * 0.12);
  const er = Math.max(1, Math.round(strokeW * 0.3));
  const clean = dilateBox(erodeBox(cand, rw, rh, er), rw, rh, er);
  for (let k = 0; k < rw * rh; k++) clean[k] &= cand[k];

  // 种子：文字框内的笔画像素
  const cur = new Uint8Array(rw * rh);
  let seeds = 0;
  for (let y = box.y0; y <= box.y1; y++) {
    for (let x = box.x0; x <= box.x1; x++) {
      const k = (y - region.y0) * rw + x - region.x0;
      if (clean[k]) { cur[k] = 1; seeds++; }
    }
  }
  if (seeds < 20) return;
  // 迭代：在已有笔画附近的候选笔画加入水印（总量有上限，防止蔓延到整张图）
  const limit = seeds * 3;
  let total = 0;
  for (let iter = 0; iter < 10 && total < limit; iter++) {
    const near = dilateBox(cur, rw, rh, reach);
    let added = 0;
    for (let k = 0; k < rw * rh; k++) {
      if (clean[k] && near[k] && !cur[k]) { cur[k] = 1; added++; }
    }
    total += added;
    if (added < 5) break;
  }
  if (total >= limit) return; // 蔓延过度，说明候选不可靠，只保留文字框本身
  // 笔画再扩一点，盖住抗锯齿边缘
  const grown = dilateBox(cur, rw, rh, ev.r + 2);
  for (let k = 0; k < rw * rh; k++) {
    if (!grown[k]) continue;
    const x = (k % rw) + region.x0, y = ((k - (k % rw)) / rw) + region.y0;
    mask[y * W + x] = 1;
  }
}

function erodeBox(m, w, h, r) {
  const inv = new Uint8Array(w * h);
  for (let k = 0; k < w * h; k++) inv[k] = m[k] ? 0 : 1;
  const d = dilateBox(inv, w, h, r);
  for (let k = 0; k < w * h; k++) d[k] = d[k] ? 0 : 1;
  return d;
}

/** 方形膨胀（可分离的滑动窗口最大值） */
function dilateBox(m, w, h, r) {
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let run = -1e9;
    for (let x = 0; x < w; x++) { if (m[y * w + x]) run = x; if (x - run <= r) tmp[y * w + x] = 1; }
    run = 1e9;
    for (let x = w - 1; x >= 0; x--) { if (m[y * w + x]) run = x; if (run - x <= r) tmp[y * w + x] = 1; }
  }
  for (let x = 0; x < w; x++) {
    let run = -1e9;
    for (let y = 0; y < h; y++) { if (tmp[y * w + x]) run = y; if (y - run <= r) out[y * w + x] = 1; }
    run = 1e9;
    for (let y = h - 1; y >= 0; y--) { if (tmp[y * w + x]) run = y; if (run - y <= r) out[y * w + x] = 1; }
  }
  return out;
}

/**
 * Logo 底板：文字框外圈如果是一块颜色均匀、边界封闭的色块（如红底白字的 Logo），
 * 把整块底板也加入蒙版。若色块一直延伸到搜索范围边缘（天空、墙面等背景），则不加入。
 */
function expandPlate(rgba, W, H, mask, b) {
  const ring = [];
  const push = (x, y) => { if (x >= 0 && y >= 0 && x < W && y < H) ring.push(y * W + x); };
  for (let x = b.x0 - 1; x <= b.x1 + 1; x++) { push(x, b.y0 - 1); push(x, b.y1 + 1); }
  for (let y = b.y0; y <= b.y1; y++) { push(b.x0 - 1, y); push(b.x1 + 1, y); }
  if (ring.length < 8) return;
  // 外圈的中位色
  const med = [0, 1, 2].map((c) => {
    const v = ring.map((i) => rgba[i * 4 + c]).sort((a, z) => a - z);
    return v[v.length >> 1];
  });
  const close = (i) => Math.abs(rgba[i * 4] - med[0]) + Math.abs(rgba[i * 4 + 1] - med[1]) + Math.abs(rgba[i * 4 + 2] - med[2]) < 60;
  let similar = 0;
  for (const i of ring) if (close(i)) similar++;
  if (similar < ring.length * 0.6) return; // 外圈不是均匀色块
  // 外圈颜色要和更外面的背景明显不同才算底板（否则就是普通背景）
  const pad = Math.round(Math.max(b.textH * 2.5, 12));
  const lim = { x0: Math.max(0, b.x0 - pad), y0: Math.max(0, b.y0 - pad), x1: Math.min(W - 1, b.x1 + pad), y1: Math.min(H - 1, b.y1 + pad) };
  const seen = new Uint8Array(W * H);
  const stack = [];
  for (const i of ring) if (close(i)) { seen[i] = 1; stack.push(i); }
  const filled = [];
  let touches = 0;
  while (stack.length) {
    const p = stack.pop();
    filled.push(p);
    const x = p % W, y = (p - x) / W;
    if (x === lim.x0 || x === lim.x1 || y === lim.y0 || y === lim.y1) {
      // 碰到搜索边界：若该边界本身就是图片边缘（Logo 贴边），不算泄漏
      const atImageEdge = (x === 0 && lim.x0 === 0) || (y === 0 && lim.y0 === 0) || (x === W - 1 && lim.x1 === W - 1) || (y === H - 1 && lim.y1 === H - 1);
      if (!atImageEdge) touches++;
      if (touches > 3) return; // 色块延伸出去了：是背景，不是底板
    }
    for (const q of [p - 1, p + 1, p - W, p + W]) {
      const qx = q % W, qy = (q - qx) / W;
      if (q < 0 || q >= W * H || seen[q] || Math.abs(qx - x) > 1) continue;
      if (qx < lim.x0 || qx > lim.x1 || qy < lim.y0 || qy > lim.y1) continue;
      seen[q] = 1;
      if (close(q)) stack.push(q);
    }
  }
  for (const p of filled) mask[p] = 1;
  // 底板内部的孔（文字已经在框里）和边缘抗锯齿：外扩 2 像素
  for (const p of filled) {
    const x = p % W, y = (p - x) / W;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < W && yy < H) mask[yy * W + xx] = 1;
    }
  }
}

function inEdgeZone(b, W, H) {
  const band = b.y1 < H * EDGE_ZONE || b.y0 > H * (1 - EDGE_ZONE) || b.x1 < W * EDGE_ZONE || b.x0 > W * (1 - EDGE_ZONE);
  // 四个角（各占 1/3）也是水印常见位置
  const C3 = 1 / 3;
  const corner = (b.x1 < W * C3 || b.x0 > W * (1 - C3)) && (b.y1 < H * C3 || b.y0 > H * (1 - C3));
  return band || corner;
}

/**
 * 自动识别水印
 * @param {'watermark'|'all'} [opts.scope] watermark：只去水印（半透明文字 + 四周边缘的文字）；all：所有文字
 * @returns {{mask: Uint8Array, regions: Array}}
 */
export async function detectWatermarks(rgba, W, H, runDet, opts = {}) {
  const scope = opts.scope || 'watermark';
  const boxes = await detectText(rgba, W, H, runDet);
  const mask = new Uint8Array(W * H);
  const regions = [];
  for (const b of boxes) {
    const ev = translucentEvidence(rgba, W, H, b);
    let kind = null;
    if (ev) kind = 'translucent';
    else if (scope === 'all' || inEdgeZone(b, W, H)) kind = 'text';
    if (!kind) continue;
    regions.push({ ...b, kind, A: ev?.A, C: ev?.C });
    for (let y = b.y0; y <= b.y1; y++) mask.fill(1, y * W + b.x0, y * W + b.x1 + 1);
    if (ev && ev.A <= 0.62 && !ev.ambiguous) growTranslucent(rgba, W, H, mask, b, ev);
    else expandPlate(rgba, W, H, mask, b);
  }
  return { mask, regions, textBoxes: boxes.length };
}
