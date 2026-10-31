/**
 * 核心算法测试（不需要下载模型，用一个“理想背景”假模型代替 LaMa）
 * 运行：node tests/core.test.mjs
 */
import { removeWatermark, MODEL_SIZE } from '../js/core.js';
import assert from 'node:assert/strict';

const W = 700, H = 420;

// 合成一张有纹理的“原图”
function makeOriginal() {
  const img = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = (y * W + x) * 4;
      const shade = x < W * 0.45 ? 70 : 190; // 明暗交界
      img[p] = shade + 10 * Math.sin(x / 9) + 4 * Math.sin(y / 5);
      img[p + 1] = shade + 8 * Math.cos(y / 11);
      img[p + 2] = shade - 20 + 6 * Math.sin((x + y) / 13);
      img[p + 3] = 255;
    }
  }
  return img;
}

// 叠加半透明白色“文字”（若干细笔画）和一块不透明 Logo
function addWatermark(orig, alpha) {
  const wm = new Uint8ClampedArray(orig);
  const cover = new Float32Array(W * H);
  for (let k = 0; k < 14; k++) {
    const x0 = 120 + k * 32;
    for (let y = 300; y < 360; y++) for (let x = x0; x < x0 + 6; x++) cover[y * W + x] = 1; // 竖笔画
    for (let y = 300; y < 306; y++) for (let x = x0; x < x0 + 22; x++) cover[y * W + x] = 1; // 横笔画
  }
  for (let i = 0; i < W * H; i++) {
    if (!cover[i]) continue;
    for (let c = 0; c < 3; c++) wm[i * 4 + c] = Math.round(orig[i * 4 + c] * (1 - alpha) + 255 * alpha);
  }
  // 不透明 Logo
  for (let y = 30; y < 90; y++) for (let x = 30; x < 160; x++) {
    const p = (y * W + x) * 4;
    wm[p] = 230; wm[p + 1] = 40; wm[p + 2] = 60;
  }
  return wm;
}

// 假模型：用“扩散填充”（从选区边界向内平滑插值）代替 LaMa，模拟一个平滑但不完美的 AI 补画
function fakeModel() {
  return async (img, msk) => {
    const S = MODEL_SIZE;
    const out = new Float32Array(3 * S * S);
    const holes = [];
    for (let i = 0; i < S * S; i++) if (msk[i]) holes.push(i);
    for (let c = 0; c < 3; c++) {
      const plane = new Float32Array(S * S);
      for (let i = 0; i < S * S; i++) plane[i] = img[c * S * S + i] * 255;
      // 初值：按列在上下最近的已知像素之间线性插值，再迭代平滑
      for (let x = 0; x < S; x++) {
        let y = 0;
        while (y < S) {
          if (!msk[y * S + x]) { y++; continue; }
          const y0 = y - 1;
          while (y < S && msk[y * S + x]) y++;
          const top = y0 >= 0 ? plane[y0 * S + x] : null;
          const bot = y < S ? plane[y * S + x] : null;
          for (let k = y0 + 1; k < y; k++) {
            const t = (k - y0) / (y - y0);
            plane[k * S + x] = top === null ? bot : bot === null ? top : top * (1 - t) + bot * t;
          }
        }
      }
      for (let it = 0; it < 100; it++) {
        for (const i of holes) {
          const x = i % S;
          let sum = 0, n = 0;
          if (x > 0) { sum += plane[i - 1]; n++; }
          if (x < S - 1) { sum += plane[i + 1]; n++; }
          if (i >= S) { sum += plane[i - S]; n++; }
          if (i < S * (S - 1)) { sum += plane[i + S]; n++; }
          plane[i] = sum / n;
        }
      }
      out.set(plane, c * S * S);
    }
    return out;
  };
}

function psnr(a, b, mask) {
  let se = 0, n = 0;
  for (let i = 0; i < W * H; i++) {
    if (mask && !mask[i]) continue;
    for (let c = 0; c < 3; c++) { se += (a[i * 4 + c] - b[i * 4 + c]) ** 2; n++; }
  }
  return 10 * Math.log10((255 * 255) / (se / n || 1e-9));
}

const orig = makeOriginal();
const wm = addWatermark(orig, 0.4);

// 用户粗略框选：文字一个框，Logo 一个框
const mask = new Uint8Array(W * H);
const rect = (x0, y0, x1, y1) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) mask[y * W + x] = 1; };
rect(110, 290, 590, 370);
rect(25, 25, 165, 95);

let info;
const out = await removeWatermark(wm, W, H, mask, fakeModel(), { onInfo: (i) => (info = i) });

// 1. 选区（含自动扩展）以外的像素必须与输入逐字节一致
const expand = 4;
let changedFar = 0;
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  let near = false;
  for (let dy = -expand; dy <= expand && !near; dy++) for (let dx = -expand; dx <= expand; dx++) {
    const yy = y + dy, xx = x + dx;
    if (yy >= 0 && yy < H && xx >= 0 && xx < W && mask[yy * W + xx]) { near = true; break; }
  }
  if (near) continue;
  const p = (y * W + x) * 4;
  if (out[p] !== wm[p] || out[p + 1] !== wm[p + 1] || out[p + 2] !== wm[p + 2] || out[p + 3] !== wm[p + 3]) changedFar++;
}
assert.equal(changedFar, 0, '选区外的像素被修改了');
console.log('✔ 选区外像素逐字节不变');

// 2. 半透明文字应被识别并还原出真实像素
assert.ok(info.translucent.length >= 1, '没有识别出半透明水印');
const textMask = new Uint8Array(W * H);
for (let y = 290; y < 370; y++) for (let x = 110; x < 590; x++) textMask[y * W + x] = 1;
const before = psnr(wm, orig, textMask);
const after = psnr(out, orig, textMask);
const tr = info.translucent.find((t) => t.C[0] === 255);
assert.ok(tr && Math.abs(tr.A - 0.4) < 0.03, `水印不透明度估计错误：${tr && tr.A}`);
console.log(`✔ 半透明文字区域 PSNR：${before.toFixed(1)} dB → ${after.toFixed(1)} dB（估计 A=${tr.A}，真实 0.4）`);
assert.ok(after > before + 10, `半透明水印还原质量不足：${after.toFixed(1)} dB`);

// 不透明 Logo 必须被 AI 补画彻底去除（不能被误判为半透明而残留）
let logoLeft = 0;
for (let y = 35; y < 85; y++) for (let x = 35; x < 155; x++) {
  const p = (y * W + x) * 4;
  if (out[p] === 230 && out[p + 1] === 40 && out[p + 2] === 60) logoLeft++;
}
assert.equal(logoLeft, 0, '不透明 Logo 有残留');
console.log('✔ 不透明 Logo 已完全去除');

// 3. 透明通道保持不变
for (let i = 3; i < out.length; i += 4) assert.equal(out[i], 255);
console.log('✔ 透明通道保持不变');

// 4. “仅 AI 重绘”模式不做半透明还原
let info2;
await removeWatermark(wm, W, H, mask, fakeModel(), { mode: 'inpaint', onInfo: (i) => (info2 = i) });
assert.equal(info2.translucent.length, 0);
console.log('✔ 仅 AI 重绘模式正常');


// 5. 自动识别：用假的文字检测器（直接给出文字位置）验证水印判定逻辑
{
  const { detectWatermarks, DET_LIMIT } = await import('../js/detect.js');
  // 假检测器：在 “文字” 位置输出高概率（收缩后的核心区域）
  const fakeDet = (rects) => async (x, w, h) => {
    const p = new Float32Array(w * h);
    const sx = w / W, sy = h / H;
    for (const [x0, y0, x1, y1] of rects) {
      for (let y = Math.round(y0 * sy); y < Math.round(y1 * sy); y++)
        for (let xx = Math.round(x0 * sx); xx < Math.round(x1 * sx); xx++) p[y * w + xx] = 0.9;
    }
    return p;
  };
  // 半透明文字（画面中下部）→ 应被识别为半透明水印
  const r1 = await detectWatermarks(wm, W, H, fakeDet([[125, 310, 585, 352]]));
  assert.ok(r1.regions.length === 1 && r1.regions[0].kind === 'translucent', '半透明文字未被识别为水印');
  console.log(`✔ 自动识别：半透明文字 → 水印（A≈${r1.regions[0].A}）`);
  // 画面正中的不透明“招牌文字” → 默认保留
  const sign = new Uint8ClampedArray(orig);
  for (let y = 190; y < 230; y++) for (let x = 260; x < 440; x++) {
    const p = (y * W + x) * 4;
    if ((x >> 3) % 2 === 0) { sign[p] = 250; sign[p + 1] = 220; sign[p + 2] = 40; }
  }
  const r2 = await detectWatermarks(sign, W, H, fakeDet([[262, 195, 438, 225]]));
  assert.equal(r2.regions.length, 0, '画面中间的正常文字被误判为水印');
  const r3 = await detectWatermarks(sign, W, H, fakeDet([[262, 195, 438, 225]]), { scope: 'all' });
  assert.equal(r3.regions.length, 1);
  console.log('✔ 自动识别：画面中间的正常文字默认保留，「所有文字」模式下去除');
  // 角落的不透明文字 → 水印
  const r4 = await detectWatermarks(wm, W, H, fakeDet([[40, 40, 150, 80]]));
  assert.ok(r4.regions.length === 1, '角落文字未被识别');
  console.log('✔ 自动识别：角落文字/Logo → 水印');
  void DET_LIMIT;
}
console.log('全部通过');
