// app.js — Main application: video input, detection loop, UI controls, multi-phone networking

import { Detector } from './detector.js';
import { MotionDetector } from './motion.js';
import { Renderer } from './renderer.js';
import { Network } from './network.js';

// --- DOM refs ---
const video = document.getElementById('videoElement');
const overlay = document.getElementById('overlayCanvas');
const statusEl = document.getElementById('status');
const statusText = document.getElementById('statusText');
const statusSpinner = statusEl?.querySelector('.spinner');
const retryBtn = document.getElementById('retryBtn');
const sourceSelect = document.getElementById('sourceSelect');
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const motionToggle = document.getElementById('motionToggle');
const motionSlider = document.getElementById('motionSlider');
const motionValue = document.getElementById('motionValue');
const connectBtn = document.getElementById('connectBtn');
const serverUrl = document.getElementById('serverUrl');
const connectionStatus = document.getElementById('connectionStatus');
const deviceList = document.getElementById('deviceList');
const logContainer = document.getElementById('activityLog');

// --- Core modules ---
const detector = new Detector();
const motionDetector = new MotionDetector();
const renderer = new Renderer(overlay);
const network = new Network();

let latestDetections = [];
let motionEnabled = true;
let networkMode = null; // null, 'source', 'viewer'

// --- Status display ---
function showStatus(msg, { spinner = false, retry = false } = {}) {
  if (!statusEl) return;
  statusEl.className = '';
  if (statusText) statusText.textContent = msg;
  if (statusSpinner) statusSpinner.className = spinner ? 'spinner' : 'spinner hidden';
  if (retryBtn) retryBtn.className = retry ? '' : 'hidden';
}
function hideStatus() {
  if (statusEl) statusEl.className = 'hidden';
}

// --- Activity log ---
function log(message) {
  if (!logContainer) return;
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = 'log-entry';

  const timeSpan = document.createElement('span');
  timeSpan.className = 'log-time';
  timeSpan.textContent = '[' + time + '] ';
  entry.appendChild(timeSpan);
  entry.appendChild(document.createTextNode(message));

  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;
  while (logContainer.children.length > 100) {
    logContainer.removeChild(logContainer.firstChild);
  }
}

// --- Video source management ---
let frameCanvas = null;
let frameCtx = null;
let stream = null;

const SAMPLES = {
  'sample-new': 'samples/new_video.mp4',
  'sample-insect': 'samples/insects_flying_forest.mp4',
  'sample-drone': 'samples/drone_flying_sky.mp4',
  'sample-bees': 'samples/bees_hive.mp4',
  'sample-dronesky': 'samples/drone_in_sky.mp4',
  'sample-faa': 'samples/drone_spotted.mp4',
};

async function setSource(source) {
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
      log('Camera started');
    } catch (e) {
      showStatus('Camera access denied. Use sample videos or upload a file.');
      log('Camera access denied: ' + e.message);
      return;
    }
  } else if (SAMPLES[source]) {
    video.src = SAMPLES[source];
    video.load();
    try {
      await video.play();
      hideStatus();
      log('Playing sample: ' + source);
    } catch {
      showStatus('Could not load video. Try a different file.');
      return;
    }
  }
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
    log('Playing uploaded file: ' + file.name);
  }).catch(() => {
    showStatus('Could not load video. Try a different file.');
  });
}

// --- Frame capture ---
function getFrame() {
  if (video.readyState < 2) return null;
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
function mergeDetections(yoloDetections, motionBlobs) {
  // Only keep relevant YOLO classes
  const relevantYolo = yoloDetections.filter(d => {
    const name = d.label;
    return name === 'airplane' || name === 'bird' || name === 'kite' || name === 'cat';
  });

  // Map YOLO labels to our classes
  for (const d of relevantYolo) {
    if (d.label === 'airplane' || d.label === 'kite') d.label = 'drone';
    else if (d.label === 'cat') d.label = 'insect';
  }

  const merged = [...relevantYolo];

  for (const blob of motionBlobs) {
    const overlaps = relevantYolo.some(d => {
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

// --- Simple tracker: snap to detections, brief persistence ---
const tracks = new Map();
let nextTrackId = 1;
const MAX_MATCH_DIST = 80;
const MAX_MISSED = 3;

function classifyBySize(w, h) {
  const area = w * h;
  if (area < 900) return 'insect';
  if (area < 90000) return 'unknown';
  return 'drone';
}

function trackDetections(detections) {
  const usedTracks = new Set();
  const usedDets = new Set();

  // Build all detection-track pairs, sort by distance, greedy assign
  const pairs = [];
  for (const det of detections) {
    const dcx = det.x + det.w / 2;
    const dcy = det.y + det.h / 2;
    for (const [id, t] of tracks) {
      const d = Math.hypot(dcx - t.cx, dcy - t.cy);
      if (d < MAX_MATCH_DIST) {
        pairs.push({ det, id, dist: d });
      }
    }
  }
  pairs.sort((a, b) => a.dist - b.dist);

  for (const p of pairs) {
    if (usedTracks.has(p.id) || usedDets.has(p.det)) continue;
    const t = tracks.get(p.id);
    const dcx = p.det.x + p.det.w / 2;
    const dcy = p.det.y + p.det.h / 2;

    // Snap directly to detection position — no smoothing, no drift
    t.cx = dcx;
    t.cy = dcy;
    t.w = p.det.w; t.h = p.det.h;
    t.confidence = p.det.confidence;
    t.source = p.det.source;
    t.missed = 0;
    t.age++;

    if (p.det.source === 'yolo' && p.det.label) {
      t.label = p.det.label;
    } else if (p.det.source === 'motion') {
      t.label = classifyBySize(p.det.w, p.det.h);
    }

    usedTracks.add(p.id);
    usedDets.add(p.det);
  }

  // New tracks for unmatched detections
  for (const det of detections) {
    if (usedDets.has(det)) continue;
    const cx = det.x + det.w / 2;
    const cy = det.y + det.h / 2;
    const label = det.label || classifyBySize(det.w, det.h);
    tracks.set(nextTrackId++, {
      cx, cy, w: det.w, h: det.h,
      label, confidence: det.confidence, source: det.source,
      missed: 0, age: 1
    });
  }

  // Remove stale tracks quickly — no coasting, no ghost markers
  for (const [id, t] of tracks) {
    if (!usedTracks.has(id)) t.missed++;
    if (t.missed > MAX_MISSED) tracks.delete(id);
  }

  const out = [];
  for (const [, t] of tracks) {
    out.push({
      x: Math.round(t.cx - t.w / 2),
      y: Math.round(t.cy - t.h / 2),
      w: t.w, h: t.h,
      label: t.label, confidence: t.confidence, source: t.source,
      missed: t.missed
    });
  }
  return out;
}

// --- Overlay sync ---
function getVideoContentRect() {
  const videoRect = video.getBoundingClientRect();
  const viewRect = video.parentElement.getBoundingClientRect();
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const elemW = videoRect.width;
  const elemH = videoRect.height;
  const scale = Math.min(elemW / vw, elemH / vh);
  const contentW = vw * scale;
  const contentH = vh * scale;
  const offsetX = (videoRect.left - viewRect.left) + (elemW - contentW) / 2;
  const offsetY = (videoRect.top - viewRect.top) + (elemH - contentH) / 2;
  return { x: offsetX, y: offsetY, w: contentW, h: contentH, scale };
}

function syncOverlay() {
  const cr = getVideoContentRect();
  const w = Math.round(cr.w);
  const h = Math.round(cr.h);
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  overlay.style.left = Math.round(cr.x) + 'px';
  overlay.style.top = Math.round(cr.y) + 'px';
  overlay.style.width = w + 'px';
  overlay.style.height = h + 'px';
}

function scaleDetections(detections) {
  const cr = getVideoContentRect();
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const sx = cr.w / vw;
  const sy = cr.h / vh;
  return detections.map(d => ({
    ...d,
    x: Math.round(d.x * sx),
    y: Math.round(d.y * sy),
    w: Math.round(d.w * sx),
    h: Math.round(d.h * sy),
  }));
}

// --- Render loop ---
let lastFrameTime = performance.now();
let fps = 0;
let frameCount = 0;

function countByClass(detections) {
  const counts = { insect: 0, drone: 0, bird: 0, unknown: 0 };
  for (const d of detections) {
    const cls = d.label || 'unknown';
    if (counts[cls] !== undefined) counts[cls]++;
    else counts.unknown++;
  }
  return counts;
}

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
  renderer.draw(scaled);
  renderer.updateStats({
    fps,
    count: latestDetections.length,
    backend: detector.getBackend(),
    inferMs: lastInferMs,
    classCounts: countByClass(latestDetections),
  });

  requestAnimationFrame(renderLoop);
}

// --- Inference loops (decoupled: motion runs fast, YOLO runs slow) ---
let lastInferMs = 0;
let lastYoloDets = [];
let yoloRunning = false;
let yoloFrameCount = 0;
const YOLO_EVERY_N = 5; // run YOLO every Nth frame

// Fast loop: motion detection every frame
function motionLoop() {
  const frame = getFrame();
  if (frame) {
    const t0 = performance.now();

    const motionBlobs = motionEnabled ? motionDetector.update(frame) : [];
    const merged = mergeDetections(lastYoloDets, motionBlobs);
    latestDetections = trackDetections(merged);
    lastInferMs = Math.round(performance.now() - t0);

    // Kick off YOLO periodically (non-blocking)
    yoloFrameCount++;
    if (yoloFrameCount >= YOLO_EVERY_N && !yoloRunning) {
      yoloFrameCount = 0;
      yoloRunning = true;
      detector.detect(frame).then(dets => {
        lastYoloDets = dets;
        yoloRunning = false;
      }).catch(() => { yoloRunning = false; });
    }

    // Send to network if connected as source
    if (network.connected && networkMode === 'source') {
      network.sendDetections(latestDetections);
    }
  }

  requestAnimationFrame(motionLoop);
}

// --- Networking ---
function setupNetwork() {
  if (!connectBtn) return;

  connectBtn.addEventListener('click', () => {
    if (network.connected) {
      network.disconnect();
      connectBtn.textContent = 'Connect';
      if (connectionStatus) connectionStatus.textContent = 'Disconnected';
      if (connectionStatus) connectionStatus.className = 'conn-status disconnected';
      log('Disconnected from server');
      return;
    }

    const url = serverUrl?.value || 'ws://localhost:8080';
    const params = new URLSearchParams(window.location.search);
    const role = params.get('role') || 'source';
    networkMode = role;

    network.connect(url, role, navigator.userAgent.includes('Mobile') ? 'Phone' : 'Desktop');

    network.onConnect = () => {
      connectBtn.textContent = 'Disconnect';
      if (connectionStatus) connectionStatus.textContent = 'Connected (' + role + ')';
      if (connectionStatus) connectionStatus.className = 'conn-status connected';
      log('Connected to server as ' + role);
    };

    network.onDisconnect = () => {
      connectBtn.textContent = 'Connect';
      if (connectionStatus) connectionStatus.textContent = 'Disconnected';
      if (connectionStatus) connectionStatus.className = 'conn-status disconnected';
      log('Disconnected from server');
    };

    network.onSources = (sources) => {
      if (!deviceList) return;
      deviceList.textContent = '';
      for (const s of sources) {
        const div = document.createElement('div');
        div.className = 'device-item';
        const dot = document.createElement('span');
        dot.className = 'device-dot connected';
        div.appendChild(dot);
        div.appendChild(document.createTextNode(' ' + s.name));
        deviceList.appendChild(div);
      }
    };
  });
}

// --- UI event handlers ---
if (sourceSelect) {
  sourceSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'upload') {
      fileInput?.click();
    } else {
      setSource(val);
    }
  });
}

if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) setSourceFromFile(e.target.files[0]);
  });
}

// Drag and drop on upload area
if (uploadArea) {
  ['dragenter', 'dragover'].forEach(ev => {
    uploadArea.addEventListener(ev, (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
  });
  ['dragleave', 'drop'].forEach(ev => {
    uploadArea.addEventListener(ev, (e) => { e.preventDefault(); uploadArea.classList.remove('dragover'); });
  });
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = Array.from(e.dataTransfer.files).find(f => f.type.startsWith('video/'));
    if (file) setSourceFromFile(file);
  });
}

if (confSlider) {
  confSlider.addEventListener('input', (e) => {
    const v = parseInt(e.target.value) / 100;
    if (confValue) confValue.textContent = v.toFixed(2);
    detector.setConfidenceThreshold(v);
  });
}

if (motionToggle) {
  motionToggle.addEventListener('change', (e) => {
    motionEnabled = e.target.checked;
    if (motionSlider) motionSlider.disabled = !motionEnabled;
  });
}

if (motionSlider) {
  motionSlider.addEventListener('input', (e) => {
    const v = parseInt(e.target.value);
    if (motionValue) motionValue.textContent = v;
    motionDetector.setThreshold(v);
  });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'KeyC') {
    const link = document.createElement('a');
    link.download = 'capture-' + Date.now() + '.png';
    link.href = overlay.toDataURL();
    link.click();
    log('Frame captured');
  }
});

// --- Init ---
async function init() {
  showStatus('Loading model...', { spinner: true });
  log('Loading YOLO model...');

  try {
    await detector.init('models/yolov11n.onnx');
    log('Model loaded (' + detector.getBackend() + ' backend)');
  } catch (e) {
    showStatus("Your browser doesn't support ONNX Runtime. Try Chrome or Edge.", { retry: true });
    log('Model load failed: ' + e.message);
    return;
  }

  showStatus('Warming up...', { spinner: true });
  await setSource('sample-insect');

  setupNetwork();

  requestAnimationFrame(renderLoop);
  requestAnimationFrame(motionLoop);
}

init();
