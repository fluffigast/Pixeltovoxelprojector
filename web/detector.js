// detector.js — YOLOv11n inference via ONNX Runtime Web
//
// Expects the global `ort` from onnxruntime-web (loaded by index.html).

import { createCanvas } from './util.js';

export const COCO_NAMES = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
  'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
  'remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
];

const MAX_CANDIDATES = 300;

/** Intersection-over-union of two [x1, y1, x2, y2] boxes. */
export function iou(a, b) {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

/**
 * Greedy non-maximum suppression. Boxes only suppress boxes of the same group.
 * @param {Array<{box:number[], score:number, group:any}>} candidates
 * @returns kept candidates, highest score first
 */
export function nms(candidates, iouThreshold) {
  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of sorted) {
    if (!kept.some(k => k.group === c.group && iou(k.box, c.box) > iouThreshold)) kept.push(c);
  }
  return kept;
}

/** Download a file as bytes, reporting progress in [0, 1] when the size is known. */
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  const total = Number(res.headers.get('Content-Length')) || 0;
  if (!res.body || !total || !onProgress) return new Uint8Array(await res.arrayBuffer());

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(received / total, 1));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

export class Detector {
  /**
   * @param {object} [options]
   * @param {Object<string,string>} [options.classMap] COCO name → output label.
   *   When given, only these classes are reported (under the mapped label).
   */
  constructor(options = {}) {
    this.confidenceThreshold = options.confidenceThreshold ?? 0.25;
    this.iouThreshold = options.iouThreshold ?? 0.45;
    this.modelWidth = options.modelWidth ?? 640;
    this.modelHeight = options.modelHeight ?? 640;
    this.classMap = options.classMap ?? null;

    this.session = null;
    this.backend = 'none';
    this.busy = false;
    this.lastInferMs = 0;

    this._modelBytes = null;
    this._canvas = null;
    this._ctx = null;
    this._input = null;
  }

  get ready() { return this.session !== null; }
  setConfidenceThreshold(v) { this.confidenceThreshold = v; }
  getBackend() { return this.backend; }

  async init(modelPath = 'models/yolov11n.onnx', { onProgress } = {}) {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime failed to load');

    // Pick the backend up front. With WebGPU the heavy work happens on the GPU;
    // with WASM, run inference in ORT's proxy worker so a slow model cannot
    // stall video playback and the motion loop. The proxy flag has to be set
    // before the first session is created, and is only used without WebGPU.
    let gpu = false;
    try { gpu = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { /* no WebGPU */ }
    ort.env.wasm.proxy = !gpu;

    // Download once; the same bytes are reused if WebGPU has to be abandoned.
    this._modelBytes = await fetchBytes(modelPath, onProgress);
    await this._createSession(gpu ? ['webgpu', 'wasm'] : ['wasm']);
  }

  async _createSession(backends) {
    let lastError;
    for (const ep of backends) {
      try {
        this.session = await ort.InferenceSession.create(this._modelBytes, { executionProviders: [ep] });
        this.backend = ep;
        if (ep === 'wasm') this._modelBytes = null;   // nothing left to fall back to
        return;
      } catch (e) {
        lastError = e;
        console.warn(`ONNX backend ${ep} failed:`, e?.message ?? e);
      }
    }
    this.session = null;
    this.backend = 'none';
    throw new Error('No ONNX Runtime backend available: ' + (lastError?.message ?? lastError));
  }

  /** Letterbox `source` into the model input (aspect preserved, gray padding). */
  _preprocess(source, srcW, srcH) {
    const mw = this.modelWidth, mh = this.modelHeight;
    if (!this._canvas) {
      this._canvas = createCanvas(mw, mh);
      this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
    }
    const scale = Math.min(mw / srcW, mh / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = (mw - newW) / 2;
    const padY = (mh - newH) / 2;

    const ctx = this._ctx;
    ctx.fillStyle = 'rgb(128,128,128)';
    ctx.fillRect(0, 0, mw, mh);
    ctx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH);
    const { data } = ctx.getImageData(0, 0, mw, mh);

    // The proxy worker takes ownership of the input buffer (it is transferred,
    // which leaves ours detached with length 0), so reallocate when that happened.
    const size = mw * mh;
    if (!this._input || this._input.length !== 3 * size) this._input = new Float32Array(3 * size);
    const input = this._input;
    for (let i = 0, j = 0; i < size; i++, j += 4) {
      input[i] = data[j] / 255;
      input[size + i] = data[j + 1] / 255;
      input[2 * size + i] = data[j + 2] / 255;
    }
    return { tensor: new ort.Tensor('float32', input, [1, 3, mh, mw]), scale, padX, padY };
  }

  /** Decode raw YOLO output ([1, 4+C, N] or [1, N, 4+C]) into boxes in source pixels. */
  _postprocess(output, scale, padX, padY, srcW, srcH) {
    const data = output.data;
    let [, d1, d2] = output.dims;
    const transposed = d1 > d2;                 // [1, N, 4+C]
    const numBoxes = transposed ? d1 : d2;
    const numFields = transposed ? d2 : d1;
    const at = transposed
      ? (field, i) => data[i * numFields + field]
      : (field, i) => data[field * numBoxes + i];

    const candidates = [];
    for (let i = 0; i < numBoxes; i++) {
      let maxScore = 0;
      let maxClass = 0;
      for (let c = 4; c < numFields; c++) {
        const score = at(c, i);
        if (score > maxScore) { maxScore = score; maxClass = c - 4; }
      }
      if (maxScore < this.confidenceThreshold) continue;

      const name = COCO_NAMES[maxClass] ?? `class_${maxClass}`;
      const label = this.classMap ? this.classMap[name] : name;
      if (!label) continue;

      const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
      const clampX = v => Math.min(srcW, Math.max(0, v));
      const clampY = v => Math.min(srcH, Math.max(0, v));
      candidates.push({
        box: [
          clampX((cx - w / 2 - padX) / scale), clampY((cy - h / 2 - padY) / scale),
          clampX((cx + w / 2 - padX) / scale), clampY((cy + h / 2 - padY) / scale),
        ],
        score: maxScore, group: label, classId: maxClass,
      });
    }

    candidates.sort((a, b) => b.score - a.score);
    if (candidates.length > MAX_CANDIDATES) candidates.length = MAX_CANDIDATES;

    return nms(candidates, this.iouThreshold).map(c => ({
      x: Math.round(c.box[0]),
      y: Math.round(c.box[1]),
      w: Math.round(c.box[2] - c.box[0]),
      h: Math.round(c.box[3] - c.box[1]),
      classId: c.classId,
      confidence: Math.round(c.score * 100) / 100,
      label: c.group,
      source: 'yolo',
    }));
  }

  /**
   * Run detection on a frame.
   * @param {CanvasImageSource} source canvas/video/bitmap holding the frame
   * @returns {Promise<Array|null>} detections in source pixels, or null if the
   *   detector is not ready or a previous call is still running.
   */
  async detect(source, srcW, srcH) {
    if (!this.session || this.busy) return null;
    this.busy = true;
    const t0 = performance.now();
    try {
      // Pixels are read synchronously here, so the caller may reuse `source`
      // as soon as this function returns its promise.
      const { tensor, scale, padX, padY } = this._preprocess(source, srcW, srcH);
      let results;
      try {
        results = await this.session.run({ [this.session.inputNames[0]]: tensor });
      } catch (e) {
        if (this.backend !== 'webgpu' || !this._modelBytes) throw e;
        // WebGPU sessions can be created fine and still fail at run time on
        // some drivers; drop to WASM for the rest of the session.
        console.warn('WebGPU inference failed, falling back to WASM:', e?.message ?? e);
        await this._createSession(['wasm']);
        return null;
      }

      const output = results[this.session.outputNames[0]];
      this.lastInferMs = Math.round(performance.now() - t0);
      return this._postprocess(output, scale, padX, padY, srcW, srcH);
    } finally {
      this.busy = false;
    }
  }
}
