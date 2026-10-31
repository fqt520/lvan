/**
 * 半透明水印“反向还原”
 *
 * 大多数平台水印（抖音号、小红书号、平铺的半透明文字/Logo）是用
 *     水印图 = (1 - α) × 原图 + α × 水印颜色
 * 叠加上去的。只要 α 不太大，水印下面的原图信息其实还在。
 * 与其让 AI “重新画”一块（人脸、文字等细节会走样），不如把这个公式反过来解：
 *     原图 = (水印图 - α × 水印颜色) / (1 - α)
 * 这样还原出来的就是真实的原图像素，而不是 AI 猜出来的内容。
 *
 * 难点在于 α 和水印颜色都未知：
 *  1. 用 AI 补画结果或形态学开运算作为“原图的粗略估计”，对每个像素投影出 α 的初值；
 *  2. 统计 α 的众数作为水印本身的不透明度 A，分别尝试白色、黑色水印，选拟合最好的；
 *  3. 用 ICM 迭代：每个像素在 [0, A] 中选择一个 α，使还原后的像素与周围像素最连贯，
 *     从而消除文字抗锯齿边缘处的残影；
 *  4. 不符合模型的像素（如过浓的投影、不透明 Logo）自动退回 AI 补画结果；
 *  5. 最后做安全校验：若还原后 α 直方图里水印的尖峰仍在，则整块使用 AI 补画结果。
 */

const CANDIDATE_STEPS = 16;
const MAX_RECOVER_ALPHA = 0.62; // 超过这个不透明度，反解会放大噪声，改用 AI 补画

/* ------------------------------------------------------------------ */
/* 背景估计：形态学开/闭运算（van Herk / Gil-Werman，O(1) 每像素）      */
/* ------------------------------------------------------------------ */
function minmax1d(src, dst, n, stride, offset, r, isMax, g, h) {
  const k = 2 * r + 1;
  const get = (i) => src[offset + i * stride];
  for (let i = 0; i < n; i++) {
    const v = get(i);
    g[i] = i % k === 0 ? v : isMax ? Math.max(g[i - 1], v) : Math.min(g[i - 1], v);
  }
  for (let i = n - 1; i >= 0; i--) {
    const v = get(i);
    h[i] = (i + 1) % k === 0 || i === n - 1 ? v : isMax ? Math.max(h[i + 1], v) : Math.min(h[i + 1], v);
  }
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - r);
    const hi = Math.min(n - 1, i + r);
    // [lo, hi] 跨越至多两个块：h[lo] 覆盖 lo..块尾，g[hi] 覆盖块头..hi
    const a = h[lo], b = g[hi];
    dst[offset + i * stride] = isMax ? Math.max(a, b) : Math.min(a, b);
  }
}

function filter2d(img, w, hgt, r, isMax) {
  const tmp = new Float32Array(img.length);
  const out = new Float32Array(img.length);
  const n = Math.max(w, hgt);
  const g = new Float32Array(n), h = new Float32Array(n);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < hgt; y++) minmax1d(img, tmp, w, 3, y * w * 3 + c, r, isMax, g, h);
    for (let x = 0; x < w; x++) minmax1d(tmp, out, hgt, w * 3, x * 3 + c, r, isMax, g, h);
  }
  return out;
}

/** 白色水印用开运算（先腐蚀后膨胀）抹掉细亮笔画；黑色水印用闭运算 */
export function morphBackground(wm, W, H, box, idx, r, white) {
  const x0 = Math.max(0, box.x0 - r), y0 = Math.max(0, box.y0 - r);
  const x1 = Math.min(W - 1, box.x1 + r), y1 = Math.min(H - 1, box.y1 + r);
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  const img = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = ((y + y0) * W + x + x0) * 4;
      const o = (y * w + x) * 3;
      img[o] = wm[p]; img[o + 1] = wm[p + 1]; img[o + 2] = wm[p + 2];
    }
  }
  const first = filter2d(img, w, h, r, !white);
  const opened = filter2d(first, w, h, r, white);
  const out = new Float32Array(idx.length * 3);
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    const x = (i % W) - x0, y = ((i - (i % W)) / W) - y0;
    const o = (y * w + x) * 3;
    out[k * 3] = opened[o]; out[k * 3 + 1] = opened[o + 1]; out[k * 3 + 2] = opened[o + 2];
  }
  return out;
}

/** 统计每个像素相对背景 bg 的 α（只统计符合“向颜色 C 混合”模型的像素） */
export function alphaHistogram(img, imgStride, bg, idx, C) {
  const hist = new Float64Array(101);
  for (let k = 0; k < idx.length; k++) {
    const p = idx[k] * imgStride;
    let num = 0, den = 0, d2 = 0;
    for (let c = 0; c < 3; c++) {
      const b = bg[k * 3 + c];
      const d = img[p + c] - b;
      num += d * (C[c] - b);
      den += (C[c] - b) * (C[c] - b);
      d2 += d * d;
    }
    if (den < 400) continue;
    const a = num / den;
    const perp2 = d2 - (num * num) / den;
    if (a > 0.04 && a < 1 && perp2 < 30 * 30) hist[Math.round(a * 100)]++;
  }
  return hist;
}

/** α=k/100 处的尖峰显著度：基线取左右两侧中较高的一侧，单调下降的纹理分布不会被误认为尖峰 */
export function prominenceAt(hist, k) {
  const s3 = hist[k - 1] + hist[k] + hist[k + 1];
  let left = 0, right = 0;
  for (let d = 5; d <= 9; d++) {
    left += hist[Math.max(0, k - d)];
    right += hist[Math.min(100, k + d)];
  }
  return s3 - (Math.max(left, right) / 5) * 3;
}

/**
 * 估计水印颜色 C、不透明度 A，以及用哪种“背景估计”最合适。
 * 背景估计候选：AI 补画结果；不同半径的形态学开/闭运算（不依赖 AI，适合细笔画文字水印，
 * 即使用户框选了很大一片区域、AI 补画已经面目全非也能正确估计）。
 */
function estimateColorAndAlpha(wm, inp, W, H, box, idx) {
  const lama = new Float32Array(idx.length * 3);
  for (let k = 0; k < idx.length; k++) {
    const p = idx[k] * 4;
    lama[k * 3] = inp[p]; lama[k * 3 + 1] = inp[p + 1]; lama[k * 3 + 2] = inp[p + 2];
  }
  const bw = box.x1 - box.x0 + 1, bh = box.y1 - box.y0 + 1;
  const radii = [3, 6, 10, 16].filter((r) => r * 4 < Math.max(bw, bh));

  // 先分别评估白色、黑色水印；颜色由所有背景估计的显著度之和投票决定
  // （单一的大半径形态学估计会在白字之间产生“变暗”的伪影，投票更稳健）
  let best = null;
  for (const C of [[255, 255, 255], [0, 0, 0]]) {
    const white = C[0] === 255;
    const bgs = [{ name: 'ai', data: lama }];
    for (const r of radii) bgs.push({ name: 'morph' + r, data: morphBackground(wm, W, H, box, idx, r, white) });
    let total = 0;
    const scored = [];
    for (const bg of bgs) {
      const hist = alphaHistogram(wm, 4, bg.data, idx, C);
      // α < 0.15 的区间被自然纹理淹没，无法可靠识别，过淡的水印交给 AI 补画
      let A = 0, hm = 0;
      for (let k = 15; k < 99; k++) {
        const prom = prominenceAt(hist, k);
        if (prom > hm) { hm = prom; A = k / 100; }
      }
      total += hm;
      scored.push({ C, A, peak: hm, bg: bg.data, bgName: bg.name });
    }
    // 显著度接近最高的几种背景估计里，优先选半径最小的形态学估计：
    // 半径越大，越多自然的细亮结构（衣服边缘、高光）会被误判为水印
    const top = Math.max(...scored.map((x) => x.peak));
    const morphs = scored.filter((x) => x.bgName !== 'ai' && x.peak >= top * 0.85);
    const bestBg = morphs.length ? morphs[0] : scored.find((x) => x.peak === top);
    if (!best || total > best.total) best = { ...bestBg, total };
  }
  return best;
}

/**
 * 用“附近未被水印覆盖的像素”来估计细笔画下的背景，从而得到更准的 α 初值。
 * 文字水印的笔画很细，旁边几个像素就是真实背景，这比 AI 补画的估计可靠。
 */
function refineAlphaLocal(wm, W, H, idx, box, al, along, free, C, A) {
  const R = 5;
  const x0 = Math.max(0, box.x0 - R - 1), y0 = Math.max(0, box.y0 - R - 1);
  const x1 = Math.min(W - 1, box.x1 + R + 1), y1 = Math.min(H - 1, box.y1 + R + 1);
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  // 可疑像素（可能被水印覆盖）及其 1 像素邻域都不算“干净背景”
  const susp = new Uint8Array(bw * bh);
  for (const i of idx) {
    if (al[i] > A * 0.15 || along[i] < -22) {
      const x = i % W, y = (i - x) / W;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx - x0, yy = y + dy - y0;
        if (xx >= 0 && yy >= 0 && xx < bw && yy < bh) susp[yy * bw + xx] = 1;
      }
    }
  }
  // 沿 8 个方向找最近的干净像素作为背景候选，各算一个 α，再连同 AI 估计取中位数。
  // 中位数对“明暗交界”很稳健：交界一侧的少数方向会给出错误的 α，但不会影响中位数。
  const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];
  const votes = [];
  const svotes = [];
  const fvotes = [];
  const med = (arr) => {
    arr.sort((u, v) => u - v);
    const m = arr.length;
    return m % 2 ? arr[(m - 1) / 2] : (arr[m / 2 - 1] + arr[m / 2]) / 2;
  };
  for (const i of idx) {
    const px = i % W, py = (i - px) / W;
    const p = i * 4;
    votes.length = 0;
    svotes.length = 0;
    fvotes.length = 0;
    free[i] = NaN;
    votes.push(al[i]);
    svotes.push(along[i]);
    for (const [dx, dy] of dirs) {
      for (let t = 1; t <= R + 3; t++) {
        const xx = px + dx * t, yy = py + dy * t;
        if (xx < x0 || yy < y0 || xx > x1 || yy > y1) break;
        if (susp[(yy - y0) * bw + (xx - x0)]) continue;
        const q = (yy * W + xx) * 4;
        let num = 0, den = 0;
        for (let c = 0; c < 3; c++) {
          const bgc = wm[q + c];
          num += (wm[p + c] - bgc) * (C[c] - bgc);
          den += (C[c] - bgc) * (C[c] - bgc);
        }
        if (den >= 400) {
          votes.push(Math.max(0, Math.min(A, num / den)));
          fvotes.push(num / den);
          svotes.push(num / Math.sqrt(den));
        }
        break;
      }
    }
    if (votes.length < 4) continue; // 周围没有足够的干净像素（大块水印），保留 AI 估计
    al[i] = med(votes);
    along[i] = med(svotes);
    if (fvotes.length >= 4) free[i] = med(fvotes); // 未截断的 α，用于重新估计 A
  }
}

/**
 * @param {Uint8ClampedArray} wm   带水印原图 RGBA
 * @param {Uint8ClampedArray} inp  AI 补画结果 RGBA（会被就地改写为最终结果）
 * @param {Uint8Array} region      需要处理的像素（1=在选区内）
 * @param {{x0,y0,x1,y1}} box      区域外接框
 * @returns {{used:boolean, A?:number, C?:number[]}}
 */
export function recoverTranslucent(wm, inp, W, H, region, box) {
  const idx = [];
  for (let y = box.y0; y <= box.y1; y++) {
    for (let x = box.x0; x <= box.x1; x++) {
      const i = y * W + x;
      if (region[i]) idx.push(i);
    }
  }
  if (idx.length < 30) return { used: false };

  const est = estimateColorAndAlpha(wm, inp, W, H, box, idx);
  if (!est || est.peak < 30) return { used: false };
  const { C, bg } = est;
  let A = est.A;
  if (A < 0.06 || A > MAX_RECOVER_ALPHA) return { used: false, A, C };

  // 初值 α；统计“与水印颜色反向”的像素（通常是水印的投影/描边）
  const n = idx.length;
  const al = new Float32Array(W * H); // 只在 region 内有意义
  const col = new Uint8Array(W * H); // 0: 主颜色 C，1: 反色 C2
  const bad = new Uint8Array(W * H);
  const core = new Uint8Array(W * H); // 明显被水印覆盖的像素
  const C2 = C[0] + C[1] + C[2] > 382 ? [0, 0, 0] : [255, 255, 255];
  const hist2 = new Float64Array(101);
  const along = new Float32Array(W * H);
  const a2s = new Float32Array(W * H);
  const aRaw = new Float32Array(W * H);
  const fits = new Uint8Array(W * H);
  let opposite = 0;
  const free = new Float32Array(W * H);
  const initAlpha = () => {
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      const p = i * 4;
      let num = 0, den = 0, num2 = 0, den2 = 0, d2 = 0;
      for (let c = 0; c < 3; c++) {
        const b = bg[k * 3 + c];
        const d = wm[p + c] - b;
        d2 += d * d;
        num += d * (C[c] - b);
        den += (C[c] - b) * (C[c] - b);
        num2 += d * (C2[c] - b);
        den2 += (C2[c] - b) * (C2[c] - b);
      }
      along[i] = den > 1 ? num / Math.sqrt(den) : 0; // 沿水印颜色方向的偏移量（像素值单位）
      const a = den > 1 ? num / den : 0;
      al[i] = Math.max(0, Math.min(A, a));
      aRaw[i] = a;
      // 颜色偏移方向必须与“朝水印颜色混合”一致（彩色 Logo 等不符合）
      const perp2 = den > 1 ? d2 - (num * num) / den : d2;
      fits[i] = perp2 < 900 + 0.1 * d2 ? 1 : 0;
      a2s[i] = den2 > 400 ? num2 / den2 : 0;
    }
    refineAlphaLocal(wm, W, H, idx, box, al, along, free, C, A);
    // 笔画核心：α 接近 A。自然纹理的 α 分布很散（常常远大于 A），据此排除
    core.fill(0);
    for (const i of idx) if (al[i] > A * 0.6 && aRaw[i] < A * 1.6 && fits[i]) core[i] = 1;
  };
  initAlpha();
  // 形态学背景在有纹理的地方偏暗/偏亮，会让 A 估计有偏差；
  // 用笔画两侧真实的干净像素重新估计一次 A，再重算一遍
  {
    const h = new Float64Array(101);
    for (const i of idx) {
      const f = free[i];
      if (core[i] && f === f && f > A * 0.4 && f < Math.min(0.99, A * 1.6)) h[Math.round(f * 100)]++;
    }
    let best = 0, A2new = 0;
    for (let k = 2; k < 99; k++) {
      const s3 = h[k - 1] + h[k] + h[k + 1];
      if (s3 > best) { best = s3; A2new = k / 100; }
    }
    if (best >= 20 && Math.abs(A2new - A) >= 0.02 && A2new >= 0.06 && A2new <= MAX_RECOVER_ALPHA) {
      A = A2new;
      initAlpha();
    }
  }

  // 安全规则：
  //  - 符合“半透明混合”模型的笔画像素太少 → 不是半透明水印；
  //  - 选区里大部分是被 AI 大幅改变、却无法用半透明模型解释的内容（例如红底白字的实心 Logo，
  //    白字部分像半透明，但红色底板不是）→ 整组交给 AI 补画，确保不透明部分被彻底去除。
  {
    let nc = 0, U = 0, Ub = 0;
    for (const i of idx) {
      nc += core[i];
      const p = i * 4;
      const d = Math.abs(wm[p] - inp[p]) + Math.abs(wm[p + 1] - inp[p + 1]) + Math.abs(wm[p + 2] - inp[p + 2]);
      if (d > 90) { U++; if (core[i]) Ub++; }
    }
    if (nc < 30) return { used: false, A, C, reason: 'no-fit' };
    if (U > 0 && (Ub < U * 0.05 || (U > n * 0.4 && Ub < U * 0.5))) return { used: false, A, C, reason: 'opaque', unexplained: +(1 - Ub / U).toFixed(2), changed: +(U / n).toFixed(2) };
  }

  // 只在水印笔画（core）及其 2 像素边带内求解 α；其余像素认定没有被水印覆盖，
  // 保持原像素不动——这样即使用户框选了一大片，框内的人脸、纹理也一个像素都不会变。
  // near：笔画周围 4 像素，水印的投影/描边只可能出现在这里
  const band = new Uint8Array(W * H);
  const near = new Uint8Array(W * H);
  for (const i of idx) {
    if (!core[i]) continue;
    const x = i % W, y = (i - x) / W;
    for (let dy = -4; dy <= 4; dy++) {
      const yy = y + dy;
      if (yy < 0 || yy >= H) continue;
      for (let dx = -4; dx <= 4; dx++) {
        const xx = x + dx;
        if (xx < 0 || xx >= W) continue;
        near[yy * W + xx] = 1;
        if (dx >= -2 && dx <= 2 && dy >= -2 && dy <= 2) band[yy * W + xx] = 1;
      }
    }
  }
  for (const i of idx) {
    if (!band[i]) al[i] = 0;
    if (near[i] && along[i] < -22) {
      opposite++;
      const a2 = a2s[i];
      if (a2 > 0.04 && a2 < 1) hist2[Math.round(a2 * 100)]++;
    }
  }

  // 反色像素足够多，说明水印带投影/描边：估计投影的不透明度 A2
  let A2 = 0;
  if (opposite > Math.max(30, est.peak * 0.5)) {
    let hm = 0;
    for (let k = 5; k < 100; k++) {
      const s3 = hist2[k - 1] + hist2[k] + hist2[k + 1];
      if (s3 > hm) { hm = s3; A2 = k / 100; }
    }
    if (A2 > MAX_RECOVER_ALPHA) {
      // 投影太浓，无法反解：这些像素交给 AI 补画
      for (const i of idx) if (near[i] && along[i] < -22) bad[i] = 1;
      A2 = 0;
    }
  }
  // 坏像素向外扩 1 像素，避免投影/描边的边缘残留
  const bad2 = new Uint8Array(bad);
  for (const i of idx) {
    if (bad[i]) continue;
    const x = i % W, y = (i - x) / W;
    for (let dy = -1; dy <= 1 && !bad2[i]; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        if (bad[yy * W + xx]) { bad2[i] = 1; break; }
      }
    }
  }

  // 候选 (α, 颜色)
  const candA = [], candC = [], candInv = [];
  for (let s2 = 0; s2 <= CANDIDATE_STEPS; s2++) {
    const a = (A * s2) / CANDIDATE_STEPS;
    candA.push(a); candC.push(C); candInv.push(1 / (1 - a));
  }
  if (A2 > 0) {
    for (let s2 = 1; s2 <= CANDIDATE_STEPS; s2++) {
      const a = (A2 * s2) / CANDIDATE_STEPS;
      candA.push(a); candC.push(C2); candInv.push(1 / (1 - a));
    }
  }
  const NC = candA.length;
  // 有半透明投影时，投影所在的外圈也需要求解
  const solve = A2 > 0 ? near : band;

  // 当前还原值（用于邻域一致性）
  const cur = new Float32Array(W * H * 3);
  const setCur = (i) => {
    const p = i * 4;
    if (!region[i]) {
      cur[i * 3] = wm[p]; cur[i * 3 + 1] = wm[p + 1]; cur[i * 3 + 2] = wm[p + 2];
    } else if (bad2[i]) {
      cur[i * 3] = inp[p]; cur[i * 3 + 1] = inp[p + 1]; cur[i * 3 + 2] = inp[p + 2];
    } else {
      const a = al[i], k = 1 / (1 - a), cc = col[i] ? C2 : C;
      for (let c = 0; c < 3; c++) cur[i * 3 + c] = (wm[p + c] - a * cc[c]) * k;
    }
  };
  // 初始化框内及外扩 1 像素
  const bx0 = Math.max(0, box.x0 - 1), by0 = Math.max(0, box.y0 - 1);
  const bx1 = Math.min(W - 1, box.x1 + 1), by1 = Math.min(H - 1, box.y1 + 1);
  for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) setCur(y * W + x);

  const nb = new Float32Array(8 * 3);
  for (let iter = 0; iter < 2; iter++) {
    for (const i of idx) {
      if (bad2[i] || !solve[i]) continue;
      const x = i % W, y = (i - x) / W;
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const j = (yy * W + xx) * 3;
          nb[k * 3] = cur[j]; nb[k * 3 + 1] = cur[j + 1]; nb[k * 3 + 2] = cur[j + 2];
          k++;
        }
      }
      const p = i * 4;
      let bestE = Infinity, bestS = 0;
      for (let s = 0; s < NC; s++) {
        const a = candA[s], f = candInv[s], cc = candC[s];
        const r0 = (wm[p] - a * cc[0]) * f;
        const r1 = (wm[p + 1] - a * cc[1]) * f;
        const r2 = (wm[p + 2] - a * cc[2]) * f;
        let e = 0;
        for (let q = 0; q < k; q++) {
          e += Math.abs(r0 - nb[q * 3]) + Math.abs(r1 - nb[q * 3 + 1]) + Math.abs(r2 - nb[q * 3 + 2]);
        }
        // 超出 0-255 的还原值说明 α 不对，加惩罚
        if (r0 < -4 || r0 > 259 || r1 < -4 || r1 > 259 || r2 < -4 || r2 > 259) e += 400 * k;
        // 先验：水印像素要么完全没被覆盖(α=0)，要么被完全覆盖(α=A)，中间值只出现在笔画边缘
        if (e < bestE) { bestE = e; bestS = s; }
      }
      al[i] = candA[bestS];
      col[i] = candC[bestS] === C ? 0 : 1;
      setCur(i);
    }
  }

  // 安全校验：还原后，α≈A 处的尖峰应当消失。若仍然存在，说明颜色/透明度估计错了，
  // 整块退回 AI 补画，绝不留下水印。
  const after = alphaHistogram(cur, 3, bg, idx, C);
  const kA = Math.round(A * 100);
  let remain = 0;
  for (let k = Math.max(1, kA - 2); k <= Math.min(99, kA + 2); k++) remain = Math.max(remain, prominenceAt(after, k));
  if (remain > est.peak * 0.35) return { used: false, A, C, rejected: true, remain, peak: est.peak };

  for (const i of idx) {
    if (bad2[i]) continue; // 保留 AI 补画结果
    const p = i * 4;
    const j = i * 3;
    inp[p] = Math.round(cur[j]);
    inp[p + 1] = Math.round(cur[j + 1]);
    inp[p + 2] = Math.round(cur[j + 2]);
  }
  return { used: true, A, C, shadowAlpha: A2, pixels: n, bg: est.bgName };
}
