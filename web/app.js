// app.js — Main application: video input, detection loop, UI controls, multi-phone networking

import { Detector } from './detector.js';
import { MotionDetector } from './motion.js';
import { Tracker } from './tracker.js';
import { Renderer } from './renderer.js';
import { Network } from './network.js';
import { createCanvas } from './util.js';

// --- Configuration ---
const MODEL_URL = 'models/yolov11n.onnx';
const DEFAULT_SOURCE = 'sample-fixed';
const SAMPLES = {
  'sample-fixed': 'samples/new_video_fixed.mp4',    // window + sky, stabilised to mimic a fixed camera
  'sample-new': 'samples/new_video.mp4',            // same scene, hand-held
  'sample-new2': 'samples/new_video2.mp4',          // hand-held, textured background (hard case)
  'sample-insect': 'samples/insects_flying_forest.mp4',
  'sample-dronex': 'samples/drone_crossing_sky.mp4', // fixed camera, drone crosses the frame
  'sample-drone': 'samples/drone_flying_sky.mp4',   // camera pans with the drone: YOLO only, no frame motion
  'sample-bees': 'samples/bees_hive.mp4',
};

// COCO has no drone or insect class. These are the classes YOLO tends to
// report for them, mapped to the labels this app uses; everything else is dropped.
const YOLO_CLASS_MAP = { airplane: 'drone', kite: 'drone', bird: 'bird', cat: 'insect' };

const MAX_PROCESS_SIDE = 1280;   // frames larger than this are downscaled before analysis
const YOLO_EVERY_N = 5;          // start a YOLO pass every Nth processed frame
const YOLO_MIN_TTL_MS = 500;     // YOLO boxes expire if no newer result replaces them
const STATS_INTERVAL_MS = 250;
const NET_SEND_INTERVAL_MS = 100;

// --- DOM refs ---
const $ = (id) => document.getElementById(id);
const video = $('videoElement');
const view = $('view');
const overlay = $('overlayCanvas');
const statusEl = $('status');
const statusText = $('statusText');
const statusSpinner = statusEl.querySelector('.loading-spinner');
const playBtn = $('playBtn');
const lnaHelp = $('lnaHelp');
const retryBtn = $('retryBtn');
const sourceSelect = $('sourceSelect');
const fileInput = $('fileInput');
const uploadArea = $('uploadArea');
const chooseFileBtn = $('chooseFileBtn');
const captureBtn = $('captureBtn');
const confSlider = $('confSlider');
const confValue = $('confValue');
const motionToggle = $('motionToggle');
const motionSlider = $('motionSlider');
const motionValue = $('motionValue');
const connectBtn = $('connectBtn');
const serverUrl = $('serverUrl');
const connectionStatus = $('connectionStatus');
const deviceList = $('deviceList');
const logContainer = $('activityLog');

// --- Core modules ---
const detector = new Detector({ classMap: YOLO_CLASS_MAP });
const motionDetector = new MotionDetector();
const tracker = new Tracker();
const renderer = new Renderer(overlay);
const network = new Network();

// --- Status display ---
function showStatus(msg, { spinner = false, play = false } = {}) {
  statusText.textContent = msg;
  statusSpinner.hidden = !spinner;
  playBtn.hidden = !play;
  statusEl.hidden = false;
}
function hideStatus() {
  statusEl.hidden = true;
}

// --- Activity log ---
function log(message) {
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  const timeSpan = document.createElement('span');
  timeSpan.className = 'log-time';
  timeSpan.textContent = '[' + new Date().toLocaleTimeString() + '] ';
  entry.append(timeSpan, message);

  logContainer.appendChild(entry);
  while (logContainer.children.length > 100) logContainer.firstChild.remove();
  logContainer.scrollTop = logContainer.scrollHeight;
}

// --- Pipeline state ---
let latestDetections = [];   // tracker output, in processing-frame pixels
let motionEnabled = true;
let motionMs = null;         // smoothed cost of frame grab + motion + tracking
let yoloDets = [];
let yoloTime = 0;
let framesSinceYolo = 0;
let procW = 0, procH = 0;    // size frames are analysed at
let frameCanvas = null;
let frameCtx = null;
let backendLabel = 'loading…';

function resetPipeline() {
  motionDetector.reset();
  tracker.reset();
  latestDetections = [];
  yoloDets = [];
  framesSinceYolo = 0;
  motionMs = null;
  renderer.clear();
}

// --- Video source management ---
let stream = null;
let objectUrl = null;
let sourceToken = 0;   // bumped on every source change so stale async work can bail out

function releaseSource() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.pause();
  video.srcObject = null;
  video.removeAttribute('src');
  video.load();
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
}

function cameraErrorMessage(e) {
  if (e?.name === 'NotFoundError' || e?.name === 'OverconstrainedError') return 'No camera found.';
  if (e?.name === 'NotReadableError') return 'The camera is in use by another app.';
  if (!navigator.mediaDevices?.getUserMedia) return 'Camera needs a secure (https) page.';
  return 'Camera access denied.';
}

/** @param {string} source key of SAMPLES, 'camera', or 'file' (with `file`) */
async function setSource(source, file) {
  const token = ++sourceToken;
  releaseSource();
  resetPipeline();

  let description;
  try {
    if (source === 'camera') {
      showStatus('Starting camera...', { spinner: true });
      const camera = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (token !== sourceToken) {          // user picked something else meanwhile
        camera.getTracks().forEach(t => t.stop());
        return;
      }
      stream = camera;
      video.srcObject = camera;
      description = 'Camera started';
    } else if (source === 'file') {
      showStatus('Loading video...', { spinner: true });
      objectUrl = URL.createObjectURL(file);
      video.src = objectUrl;
      description = 'Playing uploaded file: ' + file.name;
    } else if (SAMPLES[source]) {
      showStatus('Loading video...', { spinner: true });
      video.src = SAMPLES[source];
      description = 'Playing sample: ' + source;
    } else {
      return;
    }
    await video.play();
  } catch (e) {
    if (token !== sourceToken) return;      // superseded; the newer call owns the UI
    if (source === 'camera' && !stream) {
      showStatus(cameraErrorMessage(e) + ' Use a sample video or upload a file.');
      log('Camera unavailable: ' + (e?.message || e));
    } else if (e?.name === 'NotAllowedError') {
      showStatus('Playback needs a tap or key press.', { play: true });
    } else {
      showStatus('Could not load video. Try a different file.');
      log('Video failed to load: ' + (e?.message || e));
    }
    return;
  }
  if (token !== sourceToken) return;
  hideStatus();
  log(description);
}

// --- Frame capture ---
function getFrame() {
  if (video.readyState < 2) return null;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;

  // Analyse at native size up to MAX_PROCESS_SIDE; larger sources (1080p, 4K)
  // are downscaled so the per-frame pixel work stays bounded.
  const scale = Math.min(1, MAX_PROCESS_SIDE / Math.max(vw, vh));
  const w = Math.round(vw * scale);
  const h = Math.round(vh * scale);
  if (!frameCanvas || procW !== w || procH !== h) {
    procW = w;
    procH = h;
    frameCanvas = createCanvas(w, h);
    frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true });
    resetPipeline();
    updateLayout();
  }
  frameCtx.drawImage(video, 0, 0, w, h);
  return frameCtx.getImageData(0, 0, w, h);
}

// --- Detection merging ---
function overlaps(a, b) {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/** YOLO boxes plus every motion blob that no YOLO box already covers. */
function mergeDetections(yolo, motionBlobs) {
  if (!yolo.length) return motionBlobs;
  const merged = [...yolo];
  for (const blob of motionBlobs) {
    if (!yolo.some(d => overlaps(d, blob))) merged.push(blob);
  }
  return merged;
}

// --- Overlay layout ---
// Position of the displayed video picture inside #view, in CSS pixels.
let layout = { x: 0, y: 0, w: 0, h: 0 };

function updateLayout() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;
  const videoRect = video.getBoundingClientRect();
  const viewRect = view.getBoundingClientRect();
  const scale = Math.min(videoRect.width / vw, videoRect.height / vh);
  const w = Math.round(vw * scale);
  const h = Math.round(vh * scale);
  const x = Math.round(videoRect.left - viewRect.left + (videoRect.width - w) / 2);
  const y = Math.round(videoRect.top - viewRect.top + (videoRect.height - h) / 2);
  layout = { x, y, w, h };

  overlay.style.left = x + 'px';
  overlay.style.top = y + 'px';
  overlay.style.width = w + 'px';
  overlay.style.height = h + 'px';
  renderer.resize(w, h);
}

function render() {
  if (!procW || !layout.w) return;
  const sx = layout.w / procW;
  const sy = layout.h / procH;
  renderer.draw(latestDetections.map(d => ({
    ...d,
    x: Math.round(d.x * sx),
    y: Math.round(d.y * sy),
    w: Math.round(d.w * sx),
    h: Math.round(d.h * sy),
  })));
}

// Resizing the window or the video element clears the canvas, so redraw.
new ResizeObserver(() => { updateLayout(); render(); }).observe(video);
video.addEventListener('resize', () => { updateLayout(); render(); });

// --- Per-frame processing ---
let framesInWindow = 0;
let lastNetSend = 0;

function runYolo() {
  const token = sourceToken;
  detector.detect(frameCanvas, procW, procH).then((dets) => {
    if (!dets || token !== sourceToken) return;   // busy/not ready, or the source changed
    yoloDets = dets;
    yoloTime = performance.now();
  }).catch((e) => {
    console.warn('YOLO inference failed:', e);
  });
}

function processFrame() {
  const now = performance.now();
  const frame = getFrame();
  if (!frame) return;

  const motionBlobs = motionEnabled ? motionDetector.update(frame) : [];

  // Stop drawing YOLO boxes that are no longer being refreshed (slow model,
  // failed run) instead of leaving them frozen on screen.
  if (yoloDets.length && now - yoloTime > Math.max(YOLO_MIN_TTL_MS, detector.lastInferMs * 2.5)) {
    yoloDets = [];
  }
  latestDetections = tracker.update(mergeDetections(yoloDets, motionBlobs));

  const elapsed = performance.now() - now;
  motionMs = motionMs === null ? elapsed : motionMs * 0.9 + elapsed * 0.1;

  if (++framesSinceYolo >= YOLO_EVERY_N && detector.ready && !detector.busy) {
    framesSinceYolo = 0;
    runYolo();
  }

  if (network.connected && network.role === 'source' && now - lastNetSend >= NET_SEND_INTERVAL_MS) {
    lastNetSend = now;
    network.sendDetections(latestDetections, procW, procH);
  }

  framesInWindow++;
  render();
}

// Process each decoded video frame exactly once. Driving this from
// requestAnimationFrame alone would diff a frame against itself whenever the
// display refreshes faster than the video, which reads as "no motion".
const hasVideoFrameCallback = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
let lastMediaTime = -1;

function onFrame() {
  try {
    if (hasVideoFrameCallback || video.currentTime !== lastMediaTime) {
      lastMediaTime = video.currentTime;
      processFrame();
    }
  } finally {
    scheduleFrame();
  }
}

function scheduleFrame() {
  if (hasVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

// --- Stats ---
function countByClass(detections) {
  const counts = { insect: 0, drone: 0, bird: 0, unknown: 0 };
  for (const d of detections) {
    if (counts[d.label] !== undefined) counts[d.label]++;
    else counts.unknown++;
  }
  return counts;
}

let fps = null;
let fpsWindowStart = performance.now();

function updateStats() {
  const now = performance.now();
  if (now - fpsWindowStart >= 1000) {
    fps = Math.round(framesInWindow * 1000 / (now - fpsWindowStart));
    framesInWindow = 0;
    fpsWindowStart = now;
  }
  renderer.updateStats({
    fps,
    count: latestDetections.length,
    backend: backendLabel,
    inferMs: detector.lastInferMs,
    motionMs,
    resolution: procW ? procW + '×' + procH : '',
    classCounts: countByClass(latestDetections),
  });
}

// --- Frame capture to PNG (video + overlay) ---
function captureFrame() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  if (overlay.width && overlay.height) ctx.drawImage(overlay, 0, 0, w, h);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = 'capture-' + Date.now() + '.png';
    link.href = url;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    log('Frame captured');
  }, 'image/png');
}

// --- Networking ---
const remoteSources = new Map();   // sourceId -> { name, summary }

function renderDeviceList() {
  deviceList.textContent = '';
  for (const { name, summary } of remoteSources.values()) {
    const item = document.createElement('li');
    item.className = 'device-item';
    const dot = document.createElement('span');
    dot.className = 'device-dot connected';
    item.append(dot, ' ' + name + (summary ? ' — ' + summary : ''));
    deviceList.appendChild(item);
  }
}

// --- Local Network Access (Chrome): ws:// to localhost / LAN from an https page ---
const PRIVATE_V4 = [/^10\./, /^192\.168\./, /^172\.(1[6-9]|2\d|3[01])\./, /^169\.254\./];
/** 'loopback-network' | 'local-network' | null (public host) for a ws(s):// URL. */
function addressSpaceFor(url) {
  let host;
  try { host = new URL(url).hostname.toLowerCase(); } catch { return null; }
  host = host.replace(/^\[|\]$/g, '');
  if (host === 'localhost' || host.endsWith('.localhost') || /^127\./.test(host) || host === '::1') return 'loopback-network';
  if (host.endsWith('.local') || PRIVATE_V4.some(r => r.test(host)) || /^(fc|fd|fe80)/.test(host)) return 'local-network';
  return null;
}

/** Permission state for the address space, or 'unsupported'. Never queries the legacy combined name. */
async function lnaPermission(space) {
  if (!space || !navigator.permissions?.query) return 'unsupported';
  try {
    const status = await navigator.permissions.query({ name: space });
    status.onchange = () => showLnaHelp(space, status.state);
    return status.state;
  } catch {
    return 'unsupported';   // browser without Local Network Access permissions
  }
}

function showLnaHelp(space, state) {
  lnaHelp.hidden = state !== 'denied';
  if (state === 'denied') {
    lnaHelp.textContent = 'This site is blocked from ' + (space === 'loopback-network' ? 'apps on this device' : 'devices on your local network') +
      '. Allow "Local network access" for this site in the browser\'s site settings, then press Connect again.';
  }
}

/** Coarse form factor via UA Client Hints or feature detection, not UA-string sniffing. */
function deviceKind() {
  const mobile = navigator.userAgentData?.mobile ?? matchMedia('(pointer: coarse)').matches;
  return mobile ? 'Phone' : 'Desktop';
}

function setupNetwork() {
  // When the page is served by server.js the relay lives on the same host.
  if (location.protocol === 'http:' && location.port) serverUrl.value = 'ws://' + location.host;

  const role = new URLSearchParams(location.search).get('role') === 'viewer' ? 'viewer' : 'source';

  network.onStateChange = (state) => {
    const connected = state === 'connected';
    connectBtn.textContent = state === 'disconnected' ? 'Connect' : 'Disconnect';
    connectionStatus.className = 'conn-status ' + (connected ? 'connected' : 'disconnected');
    connectionStatus.textContent = {
      disconnected: 'Disconnected',
      connecting: 'Connecting…',
      connected: 'Connected (' + role + ')',
      reconnecting: 'Reconnecting…',
    }[state];
    if (!connected) {
      remoteSources.clear();
      renderDeviceList();
    }
    if (state !== 'connecting') {
      log({
        connected: 'Connected to server as ' + role,
        reconnecting: 'No connection to server, retrying…',
        disconnected: 'Disconnected from server',
      }[state]);
    }
  };

  network.onSources = (sources) => {
    const previous = new Map(remoteSources);
    remoteSources.clear();
    for (const s of sources) {
      remoteSources.set(s.id, { name: s.name, summary: previous.get(s.id)?.summary || '' });
    }
    renderDeviceList();
  };

  network.onDetections = (sourceId, sourceName, detections) => {
    const c = countByClass(detections);
    const summary = detections.length + ' tracked (' + c.insect + ' insect, ' + c.drone + ' drone, ' + c.bird + ' bird)';
    const entry = remoteSources.get(sourceId);
    if (entry?.summary === summary) return;
    remoteSources.set(sourceId, { name: sourceName, summary });
    renderDeviceList();
  };

  // Connecting only ever starts from this click, so a Local Network Access
  // prompt is tied to a user gesture and never fires on page load.
  connectBtn.addEventListener('click', async () => {
    if (network.state !== 'disconnected') {
      network.disconnect();
      return;
    }
    const url = serverUrl.value.trim() || 'ws://localhost:8080';
    const space = addressSpaceFor(url);
    const state = await lnaPermission(space);
    showLnaHelp(space, state);
    if (state === 'denied') { log('Local network access denied for ' + url); return; }
    try {
      network.connect(url, role, deviceKind());
    } catch (e) {
      log(e.message + (e.cause ? ' (' + e.cause.message + ')' : ''));
      return;
    }
    if (location.protocol === 'https:' && url.startsWith('ws://') && !space) {
      log('Browsers block ws:// to a public host from an https:// page — use wss://.');
    }
  });
}

// --- UI event handlers ---
sourceSelect.addEventListener('change', () => {
  const value = sourceSelect.value;
  uploadArea.classList.toggle('visible', value === 'upload');
  if (value === 'upload') fileInput.click();
  else setSource(value);
});

chooseFileBtn.addEventListener('click', () => fileInput.click());

function useFile(file) {
  if (!file) return;
  sourceSelect.value = 'upload';
  uploadArea.classList.add('visible');
  setSource('file', file);
}

fileInput.addEventListener('change', () => {
  useFile(fileInput.files[0]);
  fileInput.value = '';   // so picking the same file again still fires `change`
});

// Drag and drop onto the upload area or straight onto the video
for (const target of [uploadArea, view]) {
  for (const type of ['dragenter', 'dragover']) {
    target.addEventListener(type, (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
  }
  for (const type of ['dragleave', 'drop']) {
    target.addEventListener(type, (e) => { e.preventDefault(); uploadArea.classList.remove('dragover'); });
  }
  target.addEventListener('drop', (e) => {
    useFile(Array.from(e.dataTransfer.files).find(f => f.type.startsWith('video/')));
  });
}

function applyConfidence() {
  const v = Number(confSlider.value) / 100;
  confValue.textContent = v.toFixed(2);
  confSlider.setAttribute('aria-valuetext', v.toFixed(2));
  detector.setConfidenceThreshold(v);
}
confSlider.addEventListener('input', applyConfidence);

function applyMotionSettings() {
  motionEnabled = motionToggle.checked;
  motionSlider.disabled = !motionEnabled;
  if (!motionEnabled) motionDetector.reset();   // don't diff against a stale frame later
  const v = Number(motionSlider.value);
  motionValue.textContent = v;
  motionDetector.setThreshold(v);
}
motionToggle.addEventListener('change', applyMotionSettings);
motionSlider.addEventListener('input', applyMotionSettings);

captureBtn.addEventListener('click', captureFrame);
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;   // leave Ctrl+C etc. alone
  if (e.target.matches?.('input, select, textarea')) return;
  if (e.code === 'KeyC') captureFrame();
});

// Browsers that refuse autoplay need a gesture: a tap on the video, or the
// Play button for keyboard and switch users.
function resumePlayback() {
  if (video.paused && (video.currentSrc || video.srcObject)) {
    video.play().then(hideStatus).catch(() => {});
  }
}
view.addEventListener('click', resumePlayback);
playBtn.addEventListener('click', (e) => { e.stopPropagation(); resumePlayback(); });

retryBtn.addEventListener('click', loadModel);

// --- Init ---
async function loadModel() {
  retryBtn.hidden = true;
  backendLabel = 'loading…';
  log('Loading YOLO model...');
  try {
    await detector.init(MODEL_URL, {
      onProgress: (p) => { backendLabel = 'loading ' + Math.round(p * 100) + '%'; },
    });
    backendLabel = detector.getBackend();
    log('Model loaded (' + backendLabel + ' backend)');
  } catch (e) {
    // Motion detection does not need the model, so keep running without it.
    backendLabel = 'motion only';
    retryBtn.hidden = false;
    log('Model unavailable, running motion detection only: ' + (e?.message || e));
  }
}

function init() {
  applyConfidence();
  applyMotionSettings();
  setupNetwork();

  // Start on a sample; never open the camera or a file dialog unprompted.
  if (!SAMPLES[sourceSelect.value]) sourceSelect.value = DEFAULT_SOURCE;
  uploadArea.classList.remove('visible');
  setSource(sourceSelect.value);

  scheduleFrame();
  setInterval(updateStats, STATS_INTERVAL_MS);
  loadModel();   // in the background: video and motion detection start right away
}

init();
