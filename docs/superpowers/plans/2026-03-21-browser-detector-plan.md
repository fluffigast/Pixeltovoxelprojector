# Browser Insect & Drone Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a client-side browser app that detects insects and drones in video using YOLOv11n + motion fallback, viewable on a phone.

**Architecture:** Single HTML page loads ES modules for video input, ONNX-based YOLO detection, frame-differencing motion detection, and canvas overlay rendering. No build tools, no server-side processing. ONNX Runtime Web from CDN, model bundled as .onnx file.

**Tech Stack:** Vanilla JS (ES modules), ONNX Runtime Web 1.20+ (CDN), HTML5 Canvas, getUserMedia API

**Spec:** `docs/superpowers/specs/2026-03-21-browser-insect-drone-detector-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `web/index.html` | Create | Page shell, CDN script, module loads, UI layout |
| `web/app.js` | Create | Main loop, source switching, UI controls, merge logic |
| `web/detector.js` | Create | ONNX model loading, preprocessing, inference, NMS |
| `web/motion.js` | Create | Frame diff, connected components, compactness filter |
| `web/renderer.js` | Create | Canvas overlay: boxes, labels, stats bar |
| `web/models/yolov11n.onnx` | Create | Exported YOLOv11n model (one-time Python export) |
| `web/samples/*.mp4` | Exists | Sample videos already downloaded |

---

### Task 1: Export YOLOv11n to ONNX

**Files:**
- Create: `web/models/yolov11n.onnx`

- [ ] **Step 1: Install ultralytics in a temp venv**

```bash
cd /Users/ak/Documents/Pixeltovoxelprojector-main
python3 -m venv /tmp/yolo-export-venv
source /tmp/yolo-export-venv/bin/activate
pip install ultralytics
```

- [ ] **Step 2: Export the model**

```bash
source /tmp/yolo-export-venv/bin/activate
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', imgsz=640, simplify=True)
"
```

This downloads `yolo11n.pt` (~5MB) and exports `yolo11n.onnx` to the current directory.

- [ ] **Step 3: Move the model into web/models/**

```bash
mkdir -p web/models
mv yolo11n.onnx web/models/yolov11n.onnx
rm -f yolo11n.pt  # clean up weights file
```

- [ ] **Step 4: Verify file size**

```bash
ls -lh web/models/yolov11n.onnx
```

Expected: ~5-6MB file.

- [ ] **Step 5: Clean up temp venv**

```bash
rm -rf /tmp/yolo-export-venv
```

---

### Task 2: HTML Shell + CDN Setup

**Files:**
- Create: `web/index.html`

- [ ] **Step 1: Create index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>Insect & Drone Detector</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #111; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; height: 100dvh; display: flex; flex-direction: column; }

    #controls { display: flex; gap: 8px; padding: 8px; background: #222; align-items: center; flex-wrap: wrap; }
    #controls select, #controls input { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 4px 8px; font-size: 14px; }
    #controls label { font-size: 13px; white-space: nowrap; }
    #controls input[type="file"] { display: none; }
    #uploadBtn { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 4px 8px; font-size: 14px; cursor: pointer; }

    #view { flex: 1; position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    #videoElement { max-width: 100%; max-height: 100%; object-fit: contain; }
    #overlayCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }

    #stats { display: flex; gap: 16px; padding: 6px 12px; background: #222; font-size: 13px; font-family: monospace; }
    .stat-warn { color: #f80; }

    #status { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 18px; text-align: center; z-index: 10; }
    #status.hidden { display: none; }
    .spinner { width: 24px; height: 24px; border: 3px solid #555; border-top-color: #eee; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 8px; }
    .spinner.hidden { display: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
    #retryBtn { background: #c33; color: #fff; border: none; border-radius: 4px; padding: 6px 16px; font-size: 14px; cursor: pointer; margin-top: 8px; }
    #retryBtn.hidden { display: none; }

    #motionControls { display: flex; gap: 8px; align-items: center; }
    #motionControls input[type="range"] { width: 60px; }
  </style>
</head>
<body>
  <div id="controls">
    <select id="sourceSelect">
      <option value="sample-insect">Sample: Insects</option>
      <option value="sample-drone">Sample: Drone</option>
      <option value="camera">Camera</option>
      <option value="upload">Upload File</option>
    </select>
    <input type="file" id="fileInput" accept="video/*">
    <button id="uploadBtn" class="hidden">Choose File</button>
    <label>Conf: <input type="range" id="confSlider" min="5" max="95" value="25" step="5"><span id="confValue">0.25</span></label>
    <div id="motionControls">
      <label>Motion: <input type="range" id="motionSlider" min="2" max="20" value="8"><span id="motionValue">8</span></label>
    </div>
  </div>

  <div id="view">
    <video id="videoElement" playsinline muted loop></video>
    <canvas id="overlayCanvas"></canvas>
    <div id="status"><div class="spinner"></div><span id="statusText">Loading model...</span><button id="retryBtn" class="hidden" onclick="location.reload()">Retry</button></div>
  </div>

  <div id="stats">
    <span id="fpsDisplay">FPS: --</span>
    <span id="countDisplay">Objects: 0</span>
    <span id="backendDisplay">Backend: --</span>
    <span id="inferDisplay">Infer: --ms</span>
  </div>

  <!-- ort.all.min.js includes WebGPU + WASM + WebGL backends for fallback chain.
       Spec says ort.webgpu.min.js but that bundle lacks WASM fallback. -->
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js"></script>
  <script type="module" src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify it loads in browser**

```bash
cd /Users/ak/Documents/Pixeltovoxelprojector-main
python3 -m http.server 8080 &
# Open http://localhost:8080/web/ — should see the UI shell with "Loading model..." text
kill %1
```

---

### Task 3: Motion Detector (`motion.js`)

**Files:**
- Create: `web/motion.js`

This task builds the frame-differencing + connected-components motion detector. No ONNX dependency — pure JS. Can be tested independently.

- [ ] **Step 1: Create motion.js with MotionDetector class**

```js
// motion.js — Frame differencing + connected component motion detector
// Ported from ultimate_insect_detector.py and precise_insect_detector.py

export class MotionDetector {
  constructor(options = {}) {
    this.threshold = options.threshold ?? 8;
    this.minArea = options.minArea ?? 2;
    this.maxArea = options.maxArea ?? 500;
    this.minCompactness = options.minCompactness ?? 0.1;
    this.prevGray = null;
  }

  setThreshold(v) { this.threshold = v; }

  // Convert RGBA ImageData to grayscale Uint8Array
  static toGrayscale(imageData) {
    const { data, width, height } = imageData;
    const gray = new Uint8Array(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }
    return gray;
  }

  // Frame difference → binary mask
  _frameDiff(gray, width, height) {
    const mask = new Uint8Array(width * height);
    if (!this.prevGray) {
      this.prevGray = gray;
      return mask;
    }
    for (let i = 0; i < gray.length; i++) {
      const diff = gray[i] - this.prevGray[i];
      mask[i] = (diff > this.threshold || diff < -this.threshold) ? 1 : 0;
    }
    this.prevGray = gray;
    return mask;
  }

  // Morphological open (erode then dilate) with 2x2 kernel
  _morphOpen(mask, width, height) {
    const eroded = new Uint8Array(width * height);
    // Erode: pixel=1 only if all 4 in 2x2 block are 1
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width - 1; x++) {
        const i = y * width + x;
        eroded[i] = (mask[i] & mask[i + 1] & mask[i + width] & mask[i + width + 1]);
      }
    }
    // Dilate: pixel=1 if any in 2x2 neighborhood is 1
    const dilated = new Uint8Array(width * height);
    for (let y = 1; y < height; y++) {
      for (let x = 1; x < width; x++) {
        const i = y * width + x;
        dilated[i] = eroded[i] | eroded[i - 1] | eroded[i - width] | eroded[i - width - 1];
      }
    }
    return dilated;
  }

  // Two-pass connected component labeling with union-find
  _connectedComponents(mask, width, height) {
    const labels = new Int32Array(width * height);
    const parent = [0]; // parent[0] unused, labels start at 1
    let nextLabel = 1;

    function find(x) {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    }
    function union(a, b) {
      a = find(a); b = find(b);
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
    }

    // Pass 1: assign provisional labels
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (!mask[i]) continue;
        const above = y > 0 ? labels[i - width] : 0;
        const left = x > 0 ? labels[i - 1] : 0;
        if (above && left) {
          labels[i] = Math.min(above, left);
          union(above, left);
        } else if (above) {
          labels[i] = above;
        } else if (left) {
          labels[i] = left;
        } else {
          labels[i] = nextLabel;
          parent.push(nextLabel);
          nextLabel++;
        }
      }
    }

    // Pass 2: flatten labels
    for (let i = 0; i < labels.length; i++) {
      if (labels[i]) labels[i] = find(labels[i]);
    }

    return { labels, count: nextLabel - 1 };
  }

  // Extract bounding boxes + area + perimeter from labeled image
  _extractBlobs(labels, width, height) {
    const blobs = new Map(); // label → {minX, minY, maxX, maxY, area, perimeterPixels}

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const lbl = labels[y * width + x];
        if (!lbl) continue;
        let b = blobs.get(lbl);
        if (!b) {
          b = { minX: x, minY: y, maxX: x, maxY: y, area: 0, perim: 0 };
          blobs.set(lbl, b);
        }
        b.area++;
        if (x < b.minX) b.minX = x;
        if (x > b.maxX) b.maxX = x;
        if (y < b.minY) b.minY = y;
        if (y > b.maxY) b.maxY = y;
        // Perimeter: pixel is on edge if any 4-neighbor is background
        const i = y * width + x;
        if (x === 0 || x === width - 1 || y === 0 || y === height - 1 ||
            !labels[i - 1] || !labels[i + 1] || !labels[i - width] || !labels[i + width]) {
          b.perim++;
        }
      }
    }

    return blobs;
  }

  // Main update: returns array of motion blobs
  update(imageData) {
    const { width, height } = imageData;
    const gray = MotionDetector.toGrayscale(imageData);
    const diff = this._frameDiff(gray, width, height);
    const cleaned = this._morphOpen(diff, width, height);
    const { labels } = this._connectedComponents(cleaned, width, height);
    const rawBlobs = this._extractBlobs(labels, width, height);

    const results = [];
    for (const b of rawBlobs.values()) {
      if (b.area < this.minArea || b.area > this.maxArea) continue;
      // Compactness = 4 * pi * area / perimeter^2
      const compactness = b.perim > 0 ? (4 * Math.PI * b.area) / (b.perim * b.perim) : 0;
      if (compactness < this.minCompactness) continue;
      results.push({
        x: b.minX,
        y: b.minY,
        w: b.maxX - b.minX + 1,
        h: b.maxY - b.minY + 1,
        area: b.area,
        compactness: Math.round(compactness * 100) / 100,
        label: 'flying object',
        confidence: 0,
        source: 'motion'
      });
    }
    return results;
  }
}
```

- [ ] **Step 2: Test in browser console**

Add a temporary test in the browser console after serving:
```js
import { MotionDetector } from './motion.js';
const md = new MotionDetector();
// Create two fake frames with a small difference
const frame1 = new ImageData(new Uint8ClampedArray(100*100*4).fill(128), 100, 100);
const frame2 = new ImageData(new Uint8ClampedArray(100*100*4).fill(128), 100, 100);
// Add a 5x5 bright patch to frame2
for (let y = 10; y < 15; y++) for (let x = 10; x < 15; x++) {
  const i = (y * 100 + x) * 4;
  frame2.data[i] = 255; frame2.data[i+1] = 255; frame2.data[i+2] = 255; frame2.data[i+3] = 255;
}
md.update(frame1); // first frame, no prev → empty
const blobs = md.update(frame2); // should detect the bright patch
console.log('Blobs:', blobs); // Expect 1 blob near (10,10) with area ~25
```

---

### Task 4: YOLO Detector (`detector.js`)

**Files:**
- Create: `web/detector.js`

- [ ] **Step 1: Create detector.js**

```js
// detector.js — YOLOv11n inference via ONNX Runtime Web

// COCO class names (80 classes)
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
    // Try WebGPU first, fall back to WASM
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

  // Letterbox: resize preserving aspect ratio, pad with gray (128)
  // Returns { tensor, scale, padX, padY } for coordinate adjustment
  letterbox(imageData) {
    const { width: srcW, height: srcH } = imageData;
    const scale = Math.min(this.modelWidth / srcW, this.modelHeight / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = (this.modelWidth - newW) / 2;
    const padY = (this.modelHeight - newH) / 2;

    // Use offscreen canvas for resize
    const offscreen = new OffscreenCanvas(this.modelWidth, this.modelHeight);
    const ctx = offscreen.getContext('2d');

    // Fill with gray padding
    ctx.fillStyle = 'rgb(128,128,128)';
    ctx.fillRect(0, 0, this.modelWidth, this.modelHeight);

    // Draw source image scaled and centered
    // We need to draw from ImageData → create a temp canvas
    const srcCanvas = new OffscreenCanvas(srcW, srcH);
    const srcCtx = srcCanvas.getContext('2d');
    srcCtx.putImageData(imageData, 0, 0);
    ctx.drawImage(srcCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

    const pixels = ctx.getImageData(0, 0, this.modelWidth, this.modelHeight);

    // Convert to Float32 NCHW [1, 3, 640, 640] normalized to [0,1]
    const { data } = pixels;
    const size = this.modelWidth * this.modelHeight;
    const float32 = new Float32Array(3 * size);
    for (let i = 0; i < size; i++) {
      const j = i * 4;
      float32[i] = data[j] / 255;             // R
      float32[size + i] = data[j + 1] / 255;  // G
      float32[2 * size + i] = data[j + 2] / 255; // B
    }

    return {
      tensor: new ort.Tensor('float32', float32, [1, 3, this.modelHeight, this.modelWidth]),
      scale, padX, padY
    };
  }

  // Non-maximum suppression
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
    const feeds = { images: tensor };

    // Run inference
    const results = await this.session.run(feeds);

    // YOLOv11n output: [1, 84, 8400]
    // 84 = 4 box coords (cx, cy, w, h) + 80 class probs
    const output = results[Object.keys(results)[0]];
    const data = output.data;
    const [, numFields, numBoxes] = output.dims; // [1, 84, 8400]

    const boxes = [];
    const scores = [];
    const classIds = [];

    for (let i = 0; i < numBoxes; i++) {
      // Find best class
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

      // Get box coords (center format → corner format)
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w  = data[2 * numBoxes + i];
      const h  = data[3 * numBoxes + i];

      // Convert from letterboxed coords to original image coords
      const x1 = (cx - w / 2 - padX) / scale;
      const y1 = (cy - h / 2 - padY) / scale;
      const x2 = (cx + w / 2 - padX) / scale;
      const y2 = (cy + h / 2 - padY) / scale;

      boxes.push([x1, y1, x2, y2]);
      scores.push(maxScore);
      classIds.push(maxClass);
    }

    // NMS
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
```

- [ ] **Step 2: Verify model loads**

Serve the app and check browser console:
```bash
cd /Users/ak/Documents/Pixeltovoxelprojector-main
python3 -m http.server 8080
```
Open `http://localhost:8080/web/` — console should show `ONNX Runtime: using webgpu backend` (or `wasm`).

---

### Task 5: Overlay Renderer (`renderer.js`)

**Files:**
- Create: `web/renderer.js`

- [ ] **Step 1: Create renderer.js**

```js
// renderer.js — Canvas overlay for detection results

// Color map: class → color
function detectionColor(det) {
  if (det.source === 'motion') return '#ffcc00'; // yellow for motion-only
  const name = det.label;
  if (name === 'bird' || name === 'cat') return '#00dd00'; // green — small animals (insect proxy)
  if (name === 'airplane') return '#ff3333'; // red — aircraft/drone proxy
  return '#00aaff'; // blue — other YOLO classes
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  draw(detections, stats) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (const det of detections) {
      const color = detectionColor(det);
      const { x, y, w, h } = det;

      // Bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Label background
      const text = `${det.label} ${Math.round(det.confidence * 100)}%`;
      ctx.font = '12px monospace';
      const tm = ctx.measureText(text);
      const labelH = 16;
      const labelY = y > labelH ? y - labelH : y + h;
      ctx.fillStyle = color;
      ctx.fillRect(x, labelY, tm.width + 6, labelH);

      // Label text
      ctx.fillStyle = '#000';
      ctx.fillText(text, x + 3, labelY + 12);
    }
  }

  updateStats(stats) {
    document.getElementById('fpsDisplay').textContent = `FPS: ${stats.fps}`;
    document.getElementById('countDisplay').textContent = `Objects: ${stats.count}`;
    document.getElementById('backendDisplay').textContent = `Backend: ${stats.backend}`;
    document.getElementById('inferDisplay').textContent = `Infer: ${stats.inferMs}ms`;

    const fpsEl = document.getElementById('fpsDisplay');
    fpsEl.className = stats.fps < 5 ? 'stat-warn' : '';
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
```

---

### Task 6: Main App (`app.js`)

**Files:**
- Create: `web/app.js`

- [ ] **Step 1: Create app.js — video input + main loops + merge logic**

```js
// app.js — Main application: video input, detection loop, UI controls

import { Detector } from './detector.js';
import { MotionDetector } from './motion.js';
import { Renderer } from './renderer.js';

const video = document.getElementById('videoElement');
const overlay = document.getElementById('overlayCanvas');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('statusText');
const statusSpinner = statusEl.querySelector('.spinner');
const retryBtn = document.getElementById('retryBtn');
const sourceSelect = document.getElementById('sourceSelect');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const motionSlider = document.getElementById('motionSlider');
const motionValue = document.getElementById('motionValue');

const detector = new Detector();
const motionDetector = new MotionDetector();
const renderer = new Renderer(overlay);

let latestDetections = [];
let inferenceRunning = false;

function showStatus(msg, { spinner = false, retry = false } = {}) {
  statusEl.className = '';
  statusText.textContent = msg;
  statusSpinner.className = spinner ? 'spinner' : 'spinner hidden';
  retryBtn.className = retry ? '' : 'hidden';
}
function hideStatus() { statusEl.className = 'hidden'; }
let frameCanvas = null;
let frameCtx = null;
let stream = null; // camera stream reference

// --- Video source management ---

const SAMPLES = {
  'sample-insect': 'samples/insects_flying_forest.mp4',
  'sample-drone': 'samples/drone_flying_sky.mp4',
};

async function setSource(source) {
  // Stop any existing camera stream
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  video.src = '';
  latestDetections = [];

  if (source === 'camera') {
    showStatus('Starting camera...', { spinner: true });
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      video.srcObject = stream;
      await video.play();
      hideStatus();
    } catch (e) {
      showStatus('Camera access denied. Use sample videos or upload a file.');
      console.error('Camera error:', e);
      return;
    }
  } else if (SAMPLES[source]) {
    video.src = SAMPLES[source];
    video.load();
    try {
      await video.play();
      hideStatus();
    } catch (e) {
      showStatus('Could not load video. Try a different file.');
      return;
    }
  }
  // 'upload' is handled by file input change event
}

function setSourceFromFile(file) {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  const url = URL.createObjectURL(file);
  video.src = url;
  video.load();
  video.play().then(() => {
    hideStatus();
  }).catch(() => {
    showStatus('Could not load video. Try a different file.');
  });
}

// --- Frame capture ---

function getFrame() {
  if (video.readyState < 2) return null; // not enough data
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return null;

  if (!frameCanvas || frameCanvas.width !== w || frameCanvas.height !== h) {
    frameCanvas = new OffscreenCanvas(w, h);
    frameCtx = frameCanvas.getContext('2d');
  }
  frameCtx.drawImage(video, 0, 0, w, h);
  return frameCtx.getImageData(0, 0, w, h);
}

// --- Detection merging ---
// Motion blobs not overlapping any YOLO detection are kept as "flying object"

function mergeDetections(yoloDetections, motionBlobs) {
  const merged = [...yoloDetections];

  for (const blob of motionBlobs) {
    // Check if any YOLO box overlaps this motion blob
    const overlaps = yoloDetections.some(d => {
      const ox = Math.max(0, Math.min(d.x + d.w, blob.x + blob.w) - Math.max(d.x, blob.x));
      const oy = Math.max(0, Math.min(d.y + d.h, blob.y + blob.h) - Math.max(d.y, blob.y));
      return (ox * oy) > 0;
    });
    if (!overlaps) {
      merged.push(blob);
    }
  }
  return merged;
}

// --- Sync overlay size with video display ---

function syncOverlay() {
  const rect = video.getBoundingClientRect();
  if (overlay.width !== rect.width || overlay.height !== rect.height) {
    overlay.width = rect.width;
    overlay.height = rect.height;
  }
}

// Scale detection coords from video resolution to overlay (display) resolution
function scaleDetections(detections) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const dw = overlay.width;
  const dh = overlay.height;
  const sx = dw / vw;
  const sy = dh / vh;
  return detections.map(d => ({
    ...d,
    x: Math.round(d.x * sx),
    y: Math.round(d.y * sy),
    w: Math.round(d.w * sx),
    h: Math.round(d.h * sy),
  }));
}

// --- Render loop (runs at display refresh rate) ---

let lastFrameTime = performance.now();
let fps = 0;
let frameCount = 0;

function renderLoop() {
  frameCount++;
  const now = performance.now();
  if (now - lastFrameTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastFrameTime = now;
  }

  syncOverlay();
  const scaled = scaleDetections(latestDetections);
  renderer.draw(scaled, {});
  renderer.updateStats({
    fps,
    count: latestDetections.length,
    backend: detector.getBackend(),
    inferMs: lastInferMs,
  });

  requestAnimationFrame(renderLoop);
}

// --- Inference loop (decoupled from render, self-scheduling) ---

let lastInferMs = 0;

async function inferenceLoop() {
  if (inferenceRunning) return;
  inferenceRunning = true;

  const frame = getFrame();
  if (frame) {
    const t0 = performance.now();
    const yoloDets = await detector.detect(frame);
    lastInferMs = Math.round(performance.now() - t0);

    const motionBlobs = motionDetector.update(frame);
    latestDetections = mergeDetections(yoloDets, motionBlobs);
  }

  inferenceRunning = false;
  // Adaptive scheduling: throttle based on inference time AND measured FPS
  // Per spec: if FPS < 5, increase frame skip interval
  let delay = 0;
  if (fps < 5) delay = 100;        // severe throttle — skip more aggressively
  else if (lastInferMs > 50) delay = 50; // moderate — slight delay to reduce heat
  setTimeout(inferenceLoop, delay);
}

// --- UI event handlers ---

sourceSelect.addEventListener('change', (e) => {
  const val = e.target.value;
  if (val === 'upload') {
    uploadBtn.classList.remove('hidden');
    fileInput.click();
  } else {
    uploadBtn.classList.add('hidden');
    setSource(val);
  }
});

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) {
    setSourceFromFile(e.target.files[0]);
  }
});

confSlider.addEventListener('input', (e) => {
  const v = parseInt(e.target.value) / 100;
  confValue.textContent = v.toFixed(2);
  detector.setConfidenceThreshold(v);
});

motionSlider.addEventListener('input', (e) => {
  const v = parseInt(e.target.value);
  motionValue.textContent = v;
  motionDetector.setThreshold(v);
});

// --- Init ---

async function init() {
  showStatus('Loading model...', { spinner: true });

  try {
    await detector.init('models/yolov11n.onnx');
  } catch (e) {
    showStatus("Your browser doesn't support ONNX Runtime. Try Chrome or Edge.", { retry: true });
    console.error('Detector init failed:', e);
    return;
  }

  showStatus('Warming up...', { spinner: true });

  // Start with insect sample
  await setSource('sample-insect');

  // Start both loops
  requestAnimationFrame(renderLoop);
  inferenceLoop();
}

init();
```

---

### Task 7: Integration Test — Full Pipeline

- [ ] **Step 1: Serve and test with insect sample**

```bash
cd /Users/ak/Documents/Pixeltovoxelprojector-main
python3 -m http.server 8080
```

Open `http://localhost:8080/web/` in Chrome.

Expected:
1. "Loading model..." appears briefly
2. Insect sample video starts playing
3. Bounding boxes appear on detected objects (likely labeled "bird" or other COCO class)
4. Yellow boxes from motion detector appear on moving insects YOLO misses
5. Stats bar shows FPS, object count, and backend
6. Confidence slider adjusts detection sensitivity
7. Motion slider adjusts frame-differencing threshold

- [ ] **Step 2: Test drone sample**

Switch dropdown to "Sample: Drone".

Expected:
1. Drone video plays
2. Drone detected (likely as "airplane" or "bird")
3. Motion detector picks up drone movement

- [ ] **Step 3: Test camera**

Switch dropdown to "Camera".

Expected:
1. Permission prompt appears
2. After granting, live camera feed displays
3. Detections appear on objects in view

- [ ] **Step 4: Test on phone**

Find local IP:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Open `http://<your-ip>:8080/web/` on phone browser.

Expected:
1. Page loads, model loads
2. Sample video plays with detections
3. Camera works (rear camera via facingMode: environment)
4. UI is usable at mobile viewport size

- [ ] **Step 5: Commit**

```bash
cd /Users/ak/Documents/Pixeltovoxelprojector-main
git add web/index.html web/app.js web/detector.js web/motion.js web/renderer.js web/models/yolov11n.onnx web/samples/
git commit -m "feat: browser-based insect & drone detector (Phase 1)

Client-side detection using YOLOv11n via ONNX Runtime Web with
motion-based fallback for tiny objects. Works on phone browser.
Sample videos included for verification."
```

---

## Task Dependency Order

```
Task 1 (Export ONNX model)
  ↓
Task 2 (HTML shell)
  ↓
Task 3 (motion.js)  ←── no dependencies on Task 4/5, can run in parallel
Task 4 (detector.js) ←── depends on Task 1 (model file)
Task 5 (renderer.js) ←── no dependencies on Task 3/4
  ↓
Task 6 (app.js) ←── depends on Tasks 2-5
  ↓
Task 7 (Integration test + commit)
```

Tasks 3, 4, 5 can be implemented in parallel after Task 2 is done.
