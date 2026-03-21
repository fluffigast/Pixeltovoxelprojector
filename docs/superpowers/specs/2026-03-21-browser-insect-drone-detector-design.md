# Browser-Based Insect & Drone Detector

**Date**: 2026-03-21
**Status**: Draft
**Project**: Pixeltovoxelprojector-main

## Problem

The project has 7 overlapping Python insect detectors that only run on desktop with OpenCV. The user needs:
- Detection of both insects and drones in video
- Runs on a phone browser (client-side)
- Camera input and sample video support for verification
- Eventually: 3D voxel projection of detections

## Approach

Single-page browser app using ONNX Runtime Web + YOLOv11n for object detection. Everything runs client-side — no server. A motion-based fallback supplements the ML model until a custom insect+drone model is fine-tuned.

## Architecture

```
Browser (Phone / Desktop)
├── Video Input (getUserMedia / <video> element)
│   ├── Live camera (rear on phone, webcam on desktop)
│   └── Sample videos (insects_flying_forest.mp4, drone_flying_sky.mp4)
│
├── Detection Pipeline
│   ├── Frame capture → offscreen canvas → letterbox to 640x640 (gray padding)
│   ├── YOLOv11n via ONNX Runtime Web
│   │   ├── WebGPU backend (Chrome/Edge Android, desktop)
│   │   └── WASM SIMD fallback (Safari iOS, older browsers)
│   ├── Motion pre-filter (frame differencing in JS)
│   │   └── Labels unrecognized motion blobs as "flying object"
│   └── NMS post-processing → detection list
│
├── Overlay Renderer (canvas layer)
│   ├── Bounding boxes + labels + confidence
│   ├── Color coding: green=insect, red=drone, yellow=unknown
│   └── FPS counter, detection count
│
└── [Phase 2] Voxel View (Three.js)
    ├── Ray-casting from 2D detections to 3D voxel grid
    ├── Instanced mesh rendering
    └── Toggle between 2D overlay and 3D view
```

## Components

### 1. Video Input (`app.js`)

Responsibilities:
- Source selection: dropdown with "Camera", "Sample: Insects", "Sample: Drone", "Upload file"
- Camera: `navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })`
- Sample videos: `<video>` element loading from `samples/` directory
- Frame loop: `requestAnimationFrame` captures frames to offscreen canvas
- Letterbox resize: draw to 640x640 offscreen canvas preserving aspect ratio (gray padding), for model input

Interface:
```js
class VideoInput {
  async init(source)        // 'camera' | 'sample-insect' | 'sample-drone' | File
  getFrame() → ImageData    // current 640x640 frame
  getDisplayFrame() → HTMLVideoElement  // original resolution for overlay
  destroy()
}
```

### 2. Detection Engine (`detector.js`)

Responsibilities:
- Load YOLOv11n ONNX model on startup (~5MB)
- Select best backend: WebGPU if available, fall back to WASM SIMD
- Pre-process frame: letterbox to 640x640 (preserve aspect ratio, pad with gray 128), normalize pixels to [0,1], reshape to Float32Array [1,3,640,640] (NCHW). Adjust output coordinates back to original frame space.
- Run inference via `ort.InferenceSession`
- Post-process: decode YOLO output boxes, apply NMS, filter by confidence threshold

Interface:
```js
class Detector {
  async init()                          // load model, create session
  async detect(imageData) → Detection[] // run inference on one frame
  setConfidenceThreshold(value)         // 0.0-1.0, default 0.25
  getBackend() → string                 // 'webgpu' | 'wasm'
}

// Detection = { x, y, w, h, class, confidence, label }
```

Model details:
- YOLOv11n: 2.6M params, 6.5B FLOPs, ~5MB ONNX
- COCO pre-trained (80 classes). Relevant classes: bird (14), airplane (4)
- COCO has no insect or drone class — drones are often detected as "airplane" or "bird" at distance
- Custom fine-tuning on insect+drone datasets is a future step
- Output tensor shape: [1, 84, 8400]. Transpose to [8400, 84]. First 4 values per row = [cx, cy, w, h], remaining 80 = class probabilities. Apply confidence threshold, then NMS with IoU 0.45

### 3. Motion Fallback (`motion.js`)

Responsibilities:
- Frame differencing between consecutive frames (grayscale, via `getImageData` pixel manipulation)
- Threshold binary image + connected component labeling (scan-line two-pass algorithm on the binary mask — no OpenCV dependency)
- Filter by area: min 2px, max 500px (insects: 2-150px, drones: up to 500px). Configurable via UI sliders.
- Filter by compactness: `4 * pi * area / perimeter^2 > 0.1` (simpler than convex hull solidity, ported from `precise_insect_detector.py`)
- Label qualifying blobs as "flying object" when YOLO doesn't classify them
- This supplements YOLO for tiny objects that the model misses

Grayscale conversion: inline utility — `gray = 0.299*R + 0.587*G + 0.114*B` on the ImageData pixel array.

Connected component algorithm: Two-pass scan-line labeling with union-find on the thresholded binary image. Expected cost: <2ms for 640x480 binary image. This replaces `cv2.findContours` from the Python codebase.

Interface:
```js
class MotionDetector {
  constructor(options)             // { threshold, minArea, maxArea, minCompactness }
  update(grayFrame) → MotionBlob[]  // [{x, y, w, h, area, compactness}]
}
```

Ported from: the contour analysis logic in `ultimate_insect_detector.py` and `precise_insect_detector.py`, specifically:
- Frame differencing with threshold 4-8 (configurable, default 8)
- Morphological open (2x2 kernel for small-object preservation)
- Contour area filtering (2-500px, configurable)
- Compactness checks (>0.1)

### 4. Overlay Renderer (`renderer.js`)

Responsibilities:
- Draw bounding boxes on a transparent canvas overlaying the video
- Color by class: green=insect/small, red=drone/aircraft, yellow=unclassified motion
- Label text: class name + confidence percentage
- Stats bar: FPS, detection count, backend (WebGPU/WASM), model load time
- Confidence threshold slider (UI control)

Interface:
```js
class Renderer {
  constructor(canvas)
  draw(detections, motionBlobs, stats)  // called each frame
  clear()
}
```

### 5. Main App Loop (`app.js`)

```
init:
  videoInput.init(defaultSource)
  detector.init()
  motionDetector = new MotionDetector()
  renderer = new Renderer(overlayCanvas)

// Decoupled loops: render runs at display refresh rate, inference runs as fast as it can
// This prevents inference latency from stuttering the video display.

renderLoop (requestAnimationFrame):
  renderer.draw(latestDetections, { fps, count, backend })

inferenceLoop (setTimeout, self-scheduling):
  frame = videoInput.getFrame()
  detections = await detector.detect(frame)
  motionBlobs = motionDetector.update(grayscale(frame))
  latestDetections = mergeDetections(detections, motionBlobs)
  // Adaptive: skip frames if inference > 50ms (thermal throttling mitigation)
  scheduleNext()
```

### 6. File Layout

```
Pixeltovoxelprojector-main/
├── web/
│   ├── index.html          # Single page, loads all modules
│   ├── app.js              # Main loop, UI controls, source selection
│   ├── detector.js         # ONNX Runtime + YOLO inference
│   ├── motion.js           # Frame differencing fallback
│   ├── renderer.js         # Canvas overlay drawing
│   └── models/
│       └── yolov11n.onnx    # YOLOv11n pre-trained (~5MB, bundled in repo)
│   └── samples/            # Sample videos served from within web/ to avoid CORS
│       ├── insects_flying_forest.mp4
│       └── drone_flying_sky.mp4
└── [existing files unchanged]
```

No build step. ES modules loaded natively. ONNX Runtime loaded from CDN (all-backends bundle for WebGPU→WASM fallback chain):
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js"></script>
```
Note: `ort.all.min.js` is used instead of `ort.webgpu.min.js` because the WebGPU-only bundle lacks the WASM fallback backend. `ort` is accessed as a global object. App modules use ES module `import`/`export` for each other.

### How to Run

Serve from the project root (needed for video file loading — `file://` won't work):
```bash
cd Pixeltovoxelprojector-main
python -m http.server 8080
# Open http://localhost:8080/web/ on desktop
# Open http://<your-local-ip>:8080/web/ on phone (same WiFi network)
```

## UI Layout

```
┌──────────────────────────────────┐
│ [Source: ▼ Camera]  [Conf: 0.25] │  ← controls bar
├──────────────────────────────────┤
│                                  │
│         Video + Overlay          │  ← main view (fills screen)
│    ┌────┐                        │
│    │ 🟢 │ insect 87%             │  ← detection box example
│    └────┘                        │
│                                  │
├──────────────────────────────────┤
│ FPS: 24 | Objects: 3 | WebGPU   │  ← stats bar
└──────────────────────────────────┘
```

Mobile: controls collapse to icons. Video fills viewport. Portrait and landscape supported.

## Phasing

### Phase 1 (This build)
- Video input (camera + samples)
- YOLOv11n detection via ONNX Runtime Web
- Motion fallback for tiny objects
- Canvas overlay with boxes and labels
- Works on phone browser

### Phase 2 (Future)
- Fine-tune YOLO on insect + drone datasets (Drone-vs-Bird, Insect Detect)
- Three.js voxel projection (port from existing Python)
- Track persistence across frames (simple nearest-neighbor tracker in JS)
- Audio input for drone detection supplement (phone microphone)

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model load | < 3s | ~5MB ONNX, cached after first load |
| Inference (WebGPU) | 15-30 FPS | Android Chrome, modern phones |
| Inference (WASM) | 5-12 FPS | iOS Safari (WebGPU still experimental), older devices |
| Motion fallback | < 2ms/frame | Pure JS, lightweight |
| Total app size | < 10MB | Model + JS + samples served separately |

## Dependencies

- ONNX Runtime Web (CDN, ~2MB)
- No other external dependencies
- No build tools, no npm, no bundler

## What Gets Replaced

The 7 existing Python detectors (`ultimate_insect_detector.py`, `advanced_insect_detector.py`, `precise_insect_detector.py`, `clean_insect_analyzer.py`, `insect_detector_voxels.py`, `intelligent_insect_detector.py`, `smart_background_analyzer.py`) are not deleted but are superseded by the browser app for the primary use case. They remain as reference implementations and for desktop-only batch processing.

## Error Handling

| Condition | Behavior |
|-----------|----------|
| Camera permission denied | Show message: "Camera access denied. Use sample videos or upload a file." |
| Model fails to load | Show message with retry button. Log error to console. |
| WebGPU unavailable | Silently fall back to WASM. Show "WASM" in stats bar. |
| WASM also fails | Show message: "Your browser doesn't support ONNX Runtime. Try Chrome or Edge." |
| Video file fails to load | Show message: "Could not load video. Try a different file." |
| FPS drops below 5 | Show warning in stats bar. Increase frame skip interval. |

Loading states:
- Model loading: spinner overlay with "Loading model..." text and progress (if available from fetch)
- Camera initializing: "Starting camera..." text
- First inference: "Warming up..." (first run is slower due to shader compilation on WebGPU)

## Testing Strategy

1. Load `insects_flying_forest.mp4` — verify detections appear on flying insects
2. Load `drone_flying_sky.mp4` — verify drone is detected (likely as "airplane" or "bird" with COCO model)
3. Enable camera — verify live feed works on phone browser
4. Check FPS meets targets on both WebGPU and WASM backends
5. Test on Android Chrome and iOS Safari

## Open Questions

1. **Custom training data**: When the user wants to fine-tune for insects specifically, we'll need labeled frames — the sample videos could be annotated manually or via semi-supervised labeling
2. **Voxel phase timing**: Three.js integration deferred until Phase 1 is verified working
3. **Model source**: YOLOv11n ONNX can be exported via `ultralytics` Python package (`model.export(format='onnx')`). We need to do this once and bundle the result.
