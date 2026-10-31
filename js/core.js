/**
 * 去水印核心算法（与 UI 无关，可在浏览器和 Node 中运行）
 *
 * 为什么能做到“无痕”：
 *  1. 只裁剪水印附近的区域送进模型，能用原始分辨率就用原始分辨率，
 *     不会把整张图缩小到 512 再放大（那样会整体变糊、出现色差）。
 *  2. 选区自动向外扩展几个像素，把水印的描边、阴影、半透明光晕一起覆盖，
 *     避免残留“阴影”“白边”。
 *  3. 选区以外的像素逐字节保持原样，只把修复后的选区像素贴回原图，
 *     边缘再做几像素羽化，彻底消除接缝。
 *  4. 图片尺寸不足 512 时用镜像填充（而不是白色/黑色填充），
 *     模型看到的是自然延续的画面，不会产生“白条”。
 *  5. 细长水印（比如一整行文字）按原分辨率分块依次修复，
 *     后一块会参考前一块的修复结果，保证清晰且连续。
 */

import { recoverTranslucent } from './translucent.js';

export const MODEL_SIZE = 512;

/* ------------------------------------------------------------------ */
/* 距离变换（Felzenszwalb & Huttenlocher，精确欧氏距离平方）          */
/* ------------------------------------------------------------------ */
const INF = 1e20;

function edt1d(f, n, d, v, z) {
  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

/** 返回每个像素到最近“目标像素”(isTarget=1) 的欧氏距离平方 */
export function distanceTransform(isTarget, W, H) {
  const out = new Float32Array(W * H);
  const n = Math.max(W, H);
  const f = new Float64Array(n);
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  for (let i = 0; i < W * H; i++) out[i] = isTarget[i] ? 0 : INF;
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) f[y] = out[y * W + x];
    edt1d(f, H, d, v, z);
    for (let y = 0; y < H; y++) out[y * W + x] = d[y];
  }
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) f[x] = out[row + x];
    edt1d(f, W, d, v, z);
    for (let x = 0; x < W; x++) out[row + x] = d[x];
  }
  return out;
}

/** 圆形膨胀 */
export function dilateMask(mask, W, H, r) {
  if (r <= 0) return Uint8Array.from(mask, (m) => (m ? 1 : 0));
  const dist = distanceTransform(mask, W, H);
  const r2 = r * r + 0.5;
  const out = new Uint8Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = dist[i] <= r2 ? 1 : 0;
  return out;
}

/* ------------------------------------------------------------------ */
/* 连通区域                                                            */
/* ------------------------------------------------------------------ */
function components(mask, W, H) {
  const labels = new Int32Array(W * H).fill(-1);
  const boxes = [];
  const stack = new Int32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (!mask[i] || labels[i] !== -1) continue;
    const id = boxes.length;
    const b = { x0: W, y0: H, x1: -1, y1: -1, count: 0 };
    let sp = 0;
    stack[sp++] = i;
    labels[i] = id;
    while (sp) {
      const p = stack[--sp];
      const x = p % W;
      const y = (p - x) / W;
      b.count++;
      if (x < b.x0) b.x0 = x;
      if (x > b.x1) b.x1 = x;
      if (y < b.y0) b.y0 = y;
      if (y > b.y1) b.y1 = y;
      // 8 邻域
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const q = yy * W + xx;
          if (mask[q] && labels[q] === -1) {
            labels[q] = id;
            stack[sp++] = q;
          }
        }
      }
    }
    boxes.push(b);
  }
  return boxes;
}

/** 把距离很近的区域合并成一组，一起处理（共享上下文、减少推理次数） */
function groupBoxes(boxes, gap) {
  const groups = boxes.map((b) => ({ ...b }));
  let merged = true;
  while (merged) {
    merged = false;
    outer: for (let i = 0; i < groups.length; i++) {
      for (let j = i + 1; j < groups.length; j++) {
        const a = groups[i];
        const b = groups[j];
        const nx0 = Math.min(a.x0, b.x0);
        const ny0 = Math.min(a.y0, b.y0);
        const nx1 = Math.max(a.x1, b.x1);
        const ny1 = Math.max(a.y1, b.y1);
        const close =
          a.x0 - gap <= b.x1 && b.x0 - gap <= a.x1 &&
          a.y0 - gap <= b.y1 && b.y0 - gap <= a.y1;
        if (close) {
          groups[i] = { x0: nx0, y0: ny0, x1: nx1, y1: ny1, count: a.count + b.count };
          groups.splice(j, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  return groups;
}

/* ------------------------------------------------------------------ */
/* 处理计划：决定每次送进模型的裁剪框、缩放比例和负责回填的范围        */
/* ------------------------------------------------------------------ */
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/** 以 (cx,cy) 为中心放一个 w×h 的框，并尽量放在图片内部 */
function placeBox(cx, cy, w, h, W, H) {
  w = Math.min(w, W);
  h = Math.min(h, H);
  const x0 = clamp(Math.round(cx - w / 2), 0, W - w);
  const y0 = clamp(Math.round(cy - h / 2), 0, H - h);
  return { x: x0, y: y0, w, h };
}

export function planTiles(group, W, H, opts = {}) {
  const S = MODEL_SIZE;
  const margin = opts.margin ?? 64; // 原分辨率下每侧至少保留的上下文
  const T = S - 2 * margin; // 原分辨率下单块最多能修复的尺寸
  const bw = group.x1 - group.x0 + 1;
  const bh = group.y1 - group.y0 + 1;
  const tiles = [];

  if (bw <= T && bh <= T) {
    // 小水印：原分辨率，一次搞定，上下文尽量多
    const crop = placeBox((group.x0 + group.x1 + 1) / 2, (group.y0 + group.y1 + 1) / 2, S, S, W, H);
    tiles.push({ crop, scale: 1, own: { x0: group.x0, y0: group.y0, x1: group.x1, y1: group.y1 } });
    return tiles;
  }

  if (Math.min(bw, bh) <= T && Math.max(bw, bh) <= T * 6) {
    // 细长水印（一行文字 / 一条横幅）：沿长边按原分辨率分块
    const horizontal = bw >= bh;
    const len = horizontal ? bw : bh;
    const n = Math.ceil(len / T);
    const step = len / n;
    for (let i = 0; i < n; i++) {
      const a = Math.round(i * step);
      const b = Math.round((i + 1) * step) - 1;
      let own;
      if (horizontal) own = { x0: group.x0 + a, x1: group.x0 + b, y0: group.y0, y1: group.y1 };
      else own = { x0: group.x0, x1: group.x1, y0: group.y0 + a, y1: group.y0 + b };
      const crop = placeBox((own.x0 + own.x1 + 1) / 2, (own.y0 + own.y1 + 1) / 2, S, S, W, H);
      tiles.push({ crop, scale: 1, own });
    }
    return tiles;
  }

  // 大面积选区：需要整体语义，缩放后一次修复
  const m = Math.max(32, Math.round(Math.max(bw, bh) * 0.25));
  let side = Math.max(bw, bh) + 2 * m;
  const crop = placeBox((group.x0 + group.x1 + 1) / 2, (group.y0 + group.y1 + 1) / 2, side, side, W, H);
  const scale = Math.min(1, S / Math.max(crop.w, crop.h));
  tiles.push({ crop, scale, own: { x0: group.x0, y0: group.y0, x1: group.x1, y1: group.y1 } });
  return tiles;
}

/* ------------------------------------------------------------------ */
/* 重采样                                                              */
/* ------------------------------------------------------------------ */

/** 从 RGBA 图像中取出裁剪区域，按面积平均缩放到 tw×th，输出 RGB float(0-1) 平面格式 */
function sampleRGB(rgba, W, crop, tw, th) {
  const out = new Float32Array(3 * tw * th);
  const plane = tw * th;
  const sx = crop.w / tw;
  const sy = crop.h / th;
  for (let ty = 0; ty < th; ty++) {
    const fy0 = ty * sy;
    const fy1 = fy0 + sy;
    const iy0 = Math.floor(fy0);
    const iy1 = Math.min(crop.h, Math.ceil(fy1));
    for (let tx = 0; tx < tw; tx++) {
      const fx0 = tx * sx;
      const fx1 = fx0 + sx;
      const ix0 = Math.floor(fx0);
      const ix1 = Math.min(crop.w, Math.ceil(fx1));
      let r = 0, g = 0, b = 0, wsum = 0;
      for (let y = iy0; y < iy1; y++) {
        const wy = Math.min(fy1, y + 1) - Math.max(fy0, y);
        const row = (crop.y + y) * W + crop.x;
        for (let x = ix0; x < ix1; x++) {
          const w = wy * (Math.min(fx1, x + 1) - Math.max(fx0, x));
          const p = (row + x) * 4;
          r += rgba[p] * w;
          g += rgba[p + 1] * w;
          b += rgba[p + 2] * w;
          wsum += w;
        }
      }
      const o = ty * tw + tx;
      out[o] = r / wsum / 255;
      out[plane + o] = g / wsum / 255;
      out[2 * plane + o] = b / wsum / 255;
    }
  }
  return out;
}

/** 缩放蒙版：目标像素只要覆盖到任何一个蒙版像素就算蒙版（宁多勿少） */
function sampleMask(mask, W, crop, tw, th) {
  const out = new Uint8Array(tw * th);
  const sx = crop.w / tw;
  const sy = crop.h / th;
  for (let ty = 0; ty < th; ty++) {
    const iy0 = Math.floor(ty * sy);
    const iy1 = Math.min(crop.h, Math.ceil((ty + 1) * sy));
    for (let tx = 0; tx < tw; tx++) {
      const ix0 = Math.floor(tx * sx);
      const ix1 = Math.min(crop.w, Math.ceil((tx + 1) * sx));
      let hit = 0;
      for (let y = iy0; y < iy1 && !hit; y++) {
        const row = (crop.y + y) * W + crop.x;
        for (let x = ix0; x < ix1; x++) {
          if (mask[row + x]) { hit = 1; break; }
        }
      }
      out[ty * tw + tx] = hit;
    }
  }
  return out;
}

function reflect(i, n) {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  i = ((i % period) + period) % period;
  return i < n ? i : period - i;
}

/** 镜像填充到 S×S（避免白边/黑边） */
function padReflect(rgb, mask, tw, th, S) {
  const img = new Float32Array(3 * S * S);
  const msk = new Float32Array(S * S);
  const srcPlane = tw * th;
  const dstPlane = S * S;
  for (let y = 0; y < S; y++) {
    const sy = reflect(y, th);
    for (let x = 0; x < S; x++) {
      const sx = reflect(x, tw);
      const s = sy * tw + sx;
      const d = y * S + x;
      img[d] = rgb[s];
      img[dstPlane + d] = rgb[srcPlane + s];
      img[2 * dstPlane + d] = rgb[2 * srcPlane + s];
      msk[d] = mask[s];
    }
  }
  return { img, msk };
}

/** 双线性采样模型输出 (S×S 平面, 0-255) 中 (fx,fy) 处的颜色 */
function bilinear(out, S, tw, th, fx, fy, c) {
  fx = clamp(fx, 0, tw - 1);
  fy = clamp(fy, 0, th - 1);
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, tw - 1);
  const y1 = Math.min(y0 + 1, th - 1);
  const ax = fx - x0;
  const ay = fy - y0;
  const base = c * S * S;
  const a = out[base + y0 * S + x0];
  const b = out[base + y0 * S + x1];
  const cc = out[base + y1 * S + x0];
  const d = out[base + y1 * S + x1];
  return (a * (1 - ax) + b * ax) * (1 - ay) + (cc * (1 - ax) + d * ax) * ay;
}

/* ------------------------------------------------------------------ */
/* 主流程                                                              */
/* ------------------------------------------------------------------ */

/** 根据图片尺寸给出默认的边缘扩展像素 */
export function defaultExpand(W, H) {
  return Math.max(4, Math.round(Math.min(W, H) * 0.005));
}

/**
 * @param {Uint8ClampedArray|Uint8Array} rgba  原图 RGBA
 * @param {number} W
 * @param {number} H
 * @param {Uint8Array} mask  W*H，非 0 表示要去除的区域
 * @param {(img: Float32Array, mask: Float32Array) => Promise<Float32Array>} runModel
 *        输入 1x3x512x512 (0-1) 与 1x1x512x512 (0/1)，输出 1x3x512x512 (0-255)
 * @param {object} [opts]
 * @param {number} [opts.expand]   选区向外扩展像素
 * @param {number} [opts.feather]  边缘羽化像素
 * @param {'auto'|'inpaint'} [opts.mode]  auto: 半透明水印优先反向还原真实像素；inpaint: 全部用 AI 补画
 * @param {(done:number,total:number)=>void} [opts.onProgress]
 * @param {(info:object)=>void} [opts.onInfo]
 * @returns {Promise<Uint8ClampedArray>} 新的 RGBA（选区以外与原图逐字节一致）
 */
export async function removeWatermark(rgba, W, H, mask, runModel, opts = {}) {
  const S = MODEL_SIZE;
  const expand = Math.max(0, Math.round(opts.expand ?? defaultExpand(W, H)));
  const feather = opts.feather ?? Math.min(3, Math.max(1, expand / 2));
  const result = new Uint8ClampedArray(rgba);

  const D = dilateMask(mask, W, H, expand);

  // 羽化系数：选区内部 =1，靠近选区外边界逐渐过渡到原图
  const outside = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) outside[i] = D[i] ? 0 : 1;
  const distIn = distanceTransform(outside, W, H);
  const alpha = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (D[i]) alpha[i] = Math.min(1, Math.sqrt(distIn[i]) / feather);
  }

  const groups = groupBoxes(components(D, W, H), 24);
  const plan = [];
  for (const g of groups) plan.push(...planTiles(g, W, H));

  // pending：尚未修复的像素；对模型而言它们都是“未知”的
  const pending = new Uint8Array(D);
  let done = 0;
  opts.onProgress?.(0, plan.length);

  for (const tile of plan) {
    const { crop, scale, own } = tile;
    const tw = Math.max(1, Math.round(crop.w * scale));
    const th = Math.max(1, Math.round(crop.h * scale));

    // 这一块是否还有需要修复的像素
    let any = false;
    for (let y = own.y0; y <= own.y1 && !any; y++) {
      for (let x = own.x0; x <= own.x1; x++) if (pending[y * W + x]) { any = true; break; }
    }
    if (!any) {
      opts.onProgress?.(++done, plan.length);
      continue;
    }

    const rgb = sampleRGB(result, W, crop, tw, th);
    const m = sampleMask(pending, W, crop, tw, th);
    const { img, msk } = padReflect(rgb, m, tw, th, S);
    // 蒙版区域置零（与 LaMa 训练时的输入方式一致），模型完全看不到水印本身
    for (let i = 0; i < S * S; i++) {
      if (msk[i]) { img[i] = 0; img[S * S + i] = 0; img[2 * S * S + i] = 0; }
    }

    const out = await runModel(img, msk);

    // 回填：只写 own 范围内的待修复像素
    const sx = tw / crop.w;
    const sy = th / crop.h;
    const x0 = Math.max(own.x0, crop.x);
    const x1 = Math.min(own.x1, crop.x + crop.w - 1);
    const y0 = Math.max(own.y0, crop.y);
    const y1 = Math.min(own.y1, crop.y + crop.h - 1);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        if (!pending[i]) continue;
        let r, g, b;
        if (scale === 1) {
          const o = (y - crop.y) * S + (x - crop.x);
          r = out[o];
          g = out[S * S + o];
          b = out[2 * S * S + o];
        } else {
          const fx = (x - crop.x + 0.5) * sx - 0.5;
          const fy = (y - crop.y + 0.5) * sy - 0.5;
          r = bilinear(out, S, tw, th, fx, fy, 0);
          g = bilinear(out, S, tw, th, fx, fy, 1);
          b = bilinear(out, S, tw, th, fx, fy, 2);
        }
        const a = alpha[i];
        const p = i * 4;
        result[p] = Math.round(r * a + rgba[p] * (1 - a));
        result[p + 1] = Math.round(g * a + rgba[p + 1] * (1 - a));
        result[p + 2] = Math.round(b * a + rgba[p + 2] * (1 - a));
        // 透明通道保持原样
        pending[i] = 0;
      }
    }
    opts.onProgress?.(++done, plan.length);
  }

  const info = { regions: groups.length, modelRuns: plan.length, translucent: [] };
  if ((opts.mode ?? 'auto') === 'auto') {
    for (const g of groups) {
      const r = recoverTranslucent(rgba, result, W, H, D, g);
      if (r.used) info.translucent.push(r);
      else info.skipped = [...(info.skipped || []), r];
    }
  }
  opts.onInfo?.(info);
  return result;
}
