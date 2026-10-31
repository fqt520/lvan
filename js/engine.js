/**
 * AI 模型加载与推理（浏览器端，图片不会上传到任何服务器）
 *
 * 模型：LaMa（Large Mask Inpainting, big-lama），目前开源里去水印 / 物体消除效果最好的模型之一，
 * IOPaint(lama-cleaner) 以及很多去水印小程序用的都是它。
 * ONNX 版本来自 https://huggingface.co/Carve/LaMa-ONNX （Apache-2.0）。
 *
 * 自动识别用 PP-OCRv4 文字检测模型（4.7MB），来自 https://huggingface.co/SWHL/RapidOCR （Apache-2.0）。
 */
import { MODEL_SIZE } from './core.js';

const ORT_VERSION = '1.20.1';
const ORT_SOURCES = [
  // 自己的服务器上部署时（deploy/install.sh 会下载到这里），优先用本地文件
  new URL('../vendor/ort/', import.meta.url).href,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
  `https://registry.npmmirror.com/onnxruntime-web/${ORT_VERSION}/files/dist/`,
  `https://unpkg.com/onnxruntime-web@${ORT_VERSION}/dist/`,
];

export const LAMA = {
  file: 'lama_fp32.onnx',
  size: 208044816,
  path: 'Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',
};
export const DET = {
  file: 'ch_PP-OCRv4_det_infer.onnx',
  size: 4745517,
  path: 'SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx',
};
const MODEL_FILE = LAMA.file;

function modelSources(spec) {
  const local = new URL('../models/' + spec.file, import.meta.url).href;
  const hf = 'https://huggingface.co/' + spec.path;
  const mirror = 'https://hf-mirror.com/' + spec.path;
  let inChina = false;
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
    inChina = /Shanghai|Chongqing|Harbin|Urumqi|Hong_Kong|Macau|Taipei/.test(tz) || /^zh/i.test(navigator.language);
  } catch { /* ignore */ }
  return inChina ? [local, mirror, hf] : [local, hf, mirror];
}

/* ------------------------------------------------------------------ */
/* IndexedDB 缓存：模型只需下载一次                                     */
/* ------------------------------------------------------------------ */
const DB_NAME = 'lvan-watermark';
const STORE = 'models';

function openDB() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in globalThis)) return reject(new Error('no indexedDB'));
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function cacheGet(key) {
  try {
    const db = await openDB();
    return await new Promise((resolve) => {
      const tx = db.transaction(STORE, 'readonly');
      const r = tx.objectStore(STORE).get(key);
      r.onsuccess = () => resolve(r.result || null);
      r.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function cachePut(key, value) {
  try {
    const db = await openDB();
    await new Promise((resolve) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(value, key);
      tx.oncomplete = resolve;
      tx.onerror = resolve;
      tx.onabort = resolve;
    });
  } catch { /* 缓存失败不影响使用 */ }
}

export async function hasCachedModel() {
  const c = await cacheGet(MODEL_FILE);
  return !!(c && c.byteLength > 1e8);
}

export async function clearModelCache() {
  try {
    const db = await openDB();
    await new Promise((resolve) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).clear();
      tx.oncomplete = resolve;
      tx.onerror = resolve;
    });
  } catch { /* ignore */ }
}

/* ------------------------------------------------------------------ */
/* onnxruntime-web 加载                                                 */
/* ------------------------------------------------------------------ */
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.crossOrigin = 'anonymous';
    s.onload = resolve;
    s.onerror = () => { s.remove(); reject(new Error('load failed: ' + src)); };
    document.head.appendChild(s);
  });
}

let ortBase = null;
async function loadOrt() {
  if (globalThis.ort) return globalThis.ort;
  let lastErr;
  for (const base of ORT_SOURCES) {
    try {
      await loadScript(base + 'ort.webgpu.min.js');
      ortBase = base;
      return globalThis.ort;
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error('无法加载推理引擎（onnxruntime-web），请检查网络。' + (lastErr ? lastErr.message : ''));
}

/* ------------------------------------------------------------------ */
/* 下载模型（带进度）                                                    */
/* ------------------------------------------------------------------ */
async function fetchWithProgress(url, onProgress, signal, expected) {
  const res = await fetch(url, { signal, mode: 'cors' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || expected;
  if (!res.body || !res.body.getReader) {
    const buf = await res.arrayBuffer();
    onProgress?.(buf.byteLength, total);
    return buf;
  }
  const reader = res.body.getReader();
  const out = new Uint8Array(Math.max(total, 1));
  let received = 0;
  let chunks = null; // 长度未知 / 超出时退化为分块拼接
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!chunks && received + value.length <= out.length) {
      out.set(value, received);
    } else {
      if (!chunks) chunks = [out.subarray(0, received)];
      chunks.push(value);
    }
    received += value.length;
    onProgress?.(received, total);
  }
  if (chunks) {
    const all = new Uint8Array(received);
    let o = 0;
    for (const c of chunks) { all.set(c, o); o += c.length; }
    return all.buffer;
  }
  return out.buffer.slice(0, received);
}

async function downloadModel(spec, onStatus) {
  const minBytes = spec.size * 0.9;
  const cached = await cacheGet(spec.file);
  if (cached && cached.byteLength > minBytes) {
    onStatus?.({ stage: 'cache' });
    return cached;
  }
  const errors = [];
  for (const url of modelSources(spec)) {
    const ctrl = new AbortController();
    // 30 秒内一个字节都没收到就换下一个源
    let gotBytes = false;
    const timer = setTimeout(() => { if (!gotBytes) ctrl.abort(); }, 30000);
    try {
      onStatus?.({ stage: 'download', url, loaded: 0, total: spec.size });
      const buf = await fetchWithProgress(url, (loaded, total) => {
        gotBytes = true;
        onStatus?.({ stage: 'download', url, loaded, total });
      }, ctrl.signal, spec.size);
      clearTimeout(timer);
      if (buf.byteLength < minBytes) throw new Error('文件不完整');
      await cachePut(spec.file, buf);
      return buf;
    } catch (e) {
      clearTimeout(timer);
      errors.push(`${url}: ${e.message}`);
    }
  }
  throw new Error('模型下载失败，可以手动下载模型文件后点击“选择本地模型”。\n' + errors.join('\n'));
}

/* ------------------------------------------------------------------ */
/* 推理引擎                                                             */
/* ------------------------------------------------------------------ */
export class Engine {
  constructor() {
    this.session = null;
    this.backend = null;
    this.loading = null;
  }

  get ready() {
    return !!this.session;
  }

  /** @param {(s:object)=>void} onStatus @param {ArrayBuffer} [modelBuffer] 用户手动选择的模型 */
  load(onStatus, modelBuffer) {
    if (this.session) return Promise.resolve(this);
    if (!this.loading) {
      this.loading = this._load(onStatus, modelBuffer).catch((e) => {
        this.loading = null;
        throw e;
      });
    }
    return this.loading;
  }

  async _load(onStatus, modelBuffer) {
    onStatus?.({ stage: 'runtime' });
    const ort = await loadOrt();
    ort.env.wasm.wasmPaths = ortBase;
    const threads = globalThis.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
    ort.env.wasm.numThreads = threads;

    let buf = modelBuffer;
    if (buf) await cachePut(MODEL_FILE, buf);
    else buf = await downloadModel(LAMA, onStatus);

    onStatus?.({ stage: 'init' });
    const tryCreate = async (ep) => {
      const s = await ort.InferenceSession.create(buf, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      // 用一张空白图试跑一次，确保这个后端真的能用
      const S = MODEL_SIZE;
      const img = new ort.Tensor('float32', new Float32Array(3 * S * S).fill(0.5), [1, 3, S, S]);
      const m = new Float32Array(S * S);
      m.fill(1, S * 200, S * 210);
      const msk = new ort.Tensor('float32', m, [1, 1, S, S]);
      const out = await s.run({ image: img, mask: msk });
      const data = out[s.outputNames[0]].data;
      if (!(data.length === 3 * S * S) || !Number.isFinite(data[S * 205 + 10])) throw new Error('bad output');
      return s;
    };

    const eps = [];
    if (navigator.gpu) eps.push('webgpu');
    eps.push('wasm');
    let lastErr;
    for (const ep of eps) {
      try {
        this.session = await tryCreate(ep);
        this.backend = ep === 'webgpu' ? 'GPU 加速' : `CPU (${threads} 线程)`;
        onStatus?.({ stage: 'ready', backend: this.backend });
        return this;
      } catch (e) {
        lastErr = e;
        console.warn('backend failed', ep, e);
      }
    }
    throw new Error('模型初始化失败：' + (lastErr?.message || lastErr));
  }

  /** core.removeWatermark 所需的 runModel 回调 */
  get runModel() {
    return async (img, msk) => {
      const ort = globalThis.ort;
      const S = MODEL_SIZE;
      const out = await this.session.run({
        image: new ort.Tensor('float32', img, [1, 3, S, S]),
        mask: new ort.Tensor('float32', msk, [1, 1, S, S]),
      });
      return out[this.session.outputNames[0]].data;
    };
  }
}

/** 文字检测模型（自动识别水印用），体积小，用 CPU 推理即可 */
export class DetEngine {
  constructor() {
    this.session = null;
    this.loading = null;
  }

  get ready() {
    return !!this.session;
  }

  load(onStatus) {
    if (this.session) return Promise.resolve(this);
    if (!this.loading) {
      this.loading = (async () => {
        const ort = await loadOrt();
        ort.env.wasm.wasmPaths = ortBase;
        if (!ort.env.wasm.numThreads) {
          ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
        }
        const buf = await downloadModel(DET, onStatus);
        this.session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
        return this;
      })().catch((e) => {
        this.loading = null;
        throw e;
      });
    }
    return this.loading;
  }

  /** detect.detectWatermarks 所需的 runDet 回调 */
  get runDet() {
    return async (x, w, h) => {
      const ort = globalThis.ort;
      const out = await this.session.run({ x: new ort.Tensor('float32', x, [1, 3, h, w]) });
      return out[this.session.outputNames[0]].data;
    };
  }
}
