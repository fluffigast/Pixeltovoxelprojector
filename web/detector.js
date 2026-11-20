// detector.js — YOLOv11n inference via ONNX Runtime Web

const COCO_NAMES = [
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

export class Detector {
  constructor() {
    this.session = null;
    this.confidenceThreshold = 0.25;
    this.iouThreshold = 0.45;
    this.backend = 'unknown';
    this.modelWidth = 640;
    this.modelHeight = 640;
  }

  async init(modelPath = 'models/yolov11n.onnx') {
    const backends = ['webgpu', 'wasm'];
    for (const ep of backends) {
      try {
        const opts = { executionProviders: [ep] };
        this.session = await ort.InferenceSession.create(modelPath, opts);
        this.backend = ep;
        console.log(`ONNX Runtime: using ${ep} backend`);
        return;
      } catch (e) {
        console.warn(`Backend ${ep} failed:`, e.message);
      }
    }
    throw new Error('No ONNX Runtime backend available');
  }

  setConfidenceThreshold(v) { this.confidenceThreshold = v; }
  getBackend() { return this.backend; }

  // Letterbox resize preserving aspect ratio, pad with gray (128)
  letterbox(imageData) {
    const { width: srcW, height: srcH } = imageData;
    const scale = Math.min(this.modelWidth / srcW, this.modelHeight / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = (this.modelWidth - newW) / 2;
    const padY = (this.modelHeight - newH) / 2;

    const offscreen = new OffscreenCanvas(this.modelWidth, this.modelHeight);
    const ctx = offscreen.getContext('2d');

    ctx.fillStyle = 'rgb(128,128,128)';
    ctx.fillRect(0, 0, this.modelWidth, this.modelHeight);

    const srcCanvas = new OffscreenCanvas(srcW, srcH);
    const srcCtx = srcCanvas.getContext('2d');
    srcCtx.putImageData(imageData, 0, 0);
    ctx.drawImage(srcCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

    const pixels = ctx.getImageData(0, 0, this.modelWidth, this.modelHeight);
    const { data } = pixels;
    const size = this.modelWidth * this.modelHeight;
    const float32 = new Float32Array(3 * size);
    for (let i = 0; i < size; i++) {
      const j = i * 4;
      float32[i] = data[j] / 255;
      float32[size + i] = data[j + 1] / 255;
      float32[2 * size + i] = data[j + 2] / 255;
    }

    return {
      tensor: new ort.Tensor('float32', float32, [1, 3, this.modelHeight, this.modelWidth]),
      scale, padX, padY
    };
  }

  _nms(boxes, scores, iouThreshold) {
    const indices = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
    const keep = [];
    const suppressed = new Set();

    for (const i of indices) {
      if (suppressed.has(i)) continue;
      keep.push(i);
      for (const j of indices) {
        if (suppressed.has(j) || j === i) continue;
        if (this._iou(boxes[i], boxes[j]) > iouThreshold) {
          suppressed.add(j);
        }
      }
    }
    return keep;
  }

  _iou(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (areaA + areaB - inter + 1e-6);
  }

  async detect(imageData) {
    if (!this.session) return [];

    const { tensor, scale, padX, padY } = this.letterbox(imageData);

    // Use first input name from model (usually 'images')
    const inputName = this.session.inputNames[0];
    const feeds = { [inputName]: tensor };

    const results = await this.session.run(feeds);

    // YOLOv11n output: [1, 84, 8400]
    const output = results[Object.keys(results)[0]];
    const data = output.data;
    const [, numFields, numBoxes] = output.dims;

    const boxes = [];
    const scores = [];
    const classIds = [];

    for (let i = 0; i < numBoxes; i++) {
      let maxScore = 0;
      let maxClass = 0;
      for (let c = 4; c < numFields; c++) {
        const score = data[c * numBoxes + i];
        if (score > maxScore) {
          maxScore = score;
          maxClass = c - 4;
        }
      }

      if (maxScore < this.confidenceThreshold) continue;

      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w  = data[2 * numBoxes + i];
      const h  = data[3 * numBoxes + i];

      const x1 = (cx - w / 2 - padX) / scale;
      const y1 = (cy - h / 2 - padY) / scale;
      const x2 = (cx + w / 2 - padX) / scale;
      const y2 = (cy + h / 2 - padY) / scale;

      boxes.push([x1, y1, x2, y2]);
      scores.push(maxScore);
      classIds.push(maxClass);
    }

    const keep = this._nms(boxes, scores, this.iouThreshold);

    return keep.map(i => ({
      x: Math.round(boxes[i][0]),
      y: Math.round(boxes[i][1]),
      w: Math.round(boxes[i][2] - boxes[i][0]),
      h: Math.round(boxes[i][3] - boxes[i][1]),
      classId: classIds[i],
      confidence: Math.round(scores[i] * 100) / 100,
      label: COCO_NAMES[classIds[i]] || `class_${classIds[i]}`,
      source: 'yolo'
    }));
  }
}
