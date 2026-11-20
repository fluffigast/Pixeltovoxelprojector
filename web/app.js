// app.js — Main application: video input, detection loop, UI controls, multi-phone networking

import { Detector } from './detector.js';
import { MotionDetector } from './motion.js';
import { Tracker } from './tracker.js';
import { Renderer } from './renderer.js';
import { Network } from './network.js';
import { PeerLink, newRoomCode, normalizeRoomCode } from './peer.js';
import { Dashboard, countByClass } from './dashboard.js';
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

// What the COCO model is used for. It has no insect or drone class, so:
//  • drones are what it reports as airplane or kite, birds are birds;
//  • insects come from the motion tracker alone;
//  • big things that move for other reasons become masks — motion inside
//    their boxes is ignored, so a waving hand does not turn into "insects".
const MASK = 'mask';
const YOLO_CLASS_MAP = { airplane: 'drone', kite: 'drone', bird: 'bird' };
for (const name of ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat',
  'cat', 'dog', 'horse', 'sheep', 'cow']) YOLO_CLASS_MAP[name] = MASK;
const MASK_MARGIN = 0.08;        // masks are grown by this fraction on every side

const IS_MOBILE = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
// Frames larger than this are downscaled before analysis. Phones get a smaller
// budget: the per-pixel work is what limits their frame rate.
const MAX_PROCESS_SIDE = IS_MOBILE ? 960 : 1280;
const YOLO_EVERY_N = 5;          // start a YOLO pass every Nth processed frame
const YOLO_MIN_TTL_MS = 500;     // YOLO boxes expire if no newer result replaces them
const STATS_INTERVAL_MS = 250;

// Sharing this device's picture with a dashboard
const SHARE_INTERVAL_MS = 200;        // 5 preview frames per second
const SHARE_MAX_SIDE = 480;           // preview size, px
const SHARE_JPEG_QUALITY = 0.5;
const SHARE_MAX_BUFFERED = 1 << 20;   // skip frames while this many bytes are still queued

// --- DOM refs ---
const $ = (id) => document.getElementById(id);
const video = $('videoElement');
const view = $('view');
const overlay = $('overlayCanvas');
const statusEl = $('status');
const statusText = $('statusText');
const statusSpinner = statusEl.querySelector('.spinner');
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
const dashboardEl = $('dashboard');
const roomInput = $('roomInput');
const joinBtn = $('joinBtn');
const dashboardBtn = $('dashboardBtn');
const roomInfo = $('roomInfo');
const roomCodeEl = $('roomCode');
const copyLinkBtn = $('copyLinkBtn');
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
const dashboard = new Dashboard(dashboardEl);
const params = new URLSearchParams(location.search);
// Two ways to link devices, with the same interface: room codes (device to
// device, works from the hosted page) and the WebSocket relay in server.js.
const peerLink = new PeerLink({ broker: params.get('peer') });
const relayLink = new Network();
let link = null;             // whichever of the two is in use
let dashboardMode = false;   // this device is showing other devices' feeds

// --- Status display ---
function showStatus(msg, { spinner = false } = {}) {
  statusText.textContent = msg;
  statusSpinner.hidden = !spinner;
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
      showStatus('Tap the video to start playback.');
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

/** Model detections plus every motion blob that no model box already covers. */
function mergeDetections(yolo, motionBlobs) {
  if (!yolo.length) return motionBlobs;
  const merged = yolo.filter(d => d.label !== MASK);
  for (const blob of motionBlobs) {
    if (!yolo.some(d => overlaps(d, blob))) merged.push(blob);
  }
  return merged;
}

/** Regions where motion is explained by something the model recognised. */
function maskRegions(yolo) {
  const masks = [];
  for (const d of yolo) {
    if (d.label !== MASK) continue;
    const mx = d.w * MASK_MARGIN, my = d.h * MASK_MARGIN;
    masks.push({ x: d.x - mx, y: d.y - my, w: d.w + 2 * mx, h: d.h + 2 * my });
  }
  return masks;
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
    trail: d.trail && d.trail.map(p => [p[0] * sx, p[1] * sy]),
  })));
}

// Resizing the window or the video element clears the canvas, so redraw.
new ResizeObserver(() => { updateLayout(); render(); }).observe(video);
video.addEventListener('resize', () => { updateLayout(); render(); });

// --- Per-frame processing ---
let framesInWindow = 0;
let lastShare = 0;
let shareCanvas = null;
let shareCtx = null;

/** Send a small JPEG of the current frame plus its detections to the dashboard. */
function shareFrame() {
  const scale = Math.min(1, SHARE_MAX_SIDE / Math.max(procW, procH));
  const w = Math.round(procW * scale);
  const h = Math.round(procH * scale);
  if (!shareCanvas || shareCanvas.width !== w || shareCanvas.height !== h) {
    shareCanvas = document.createElement('canvas');   // toDataURL needs a real canvas
    shareCanvas.width = w;
    shareCanvas.height = h;
    shareCtx = shareCanvas.getContext('2d');
  }
  shareCtx.drawImage(frameCanvas, 0, 0, w, h);
  const jpeg = shareCanvas.toDataURL('image/jpeg', SHARE_JPEG_QUALITY);
  const detections = latestDetections.map(({ x, y, w: dw, h: dh, label, confidence, missed, trail }) => ({
    x, y, w: dw, h: dh, label, confidence, missed,
    trail: trail && trail.slice(-8).map(p => [Math.round(p[0]), Math.round(p[1])]),
  }));
  link.sendFrame(jpeg, detections, procW, procH);
}

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
  if (dashboardMode) return;
  const now = performance.now();
  const frame = getFrame();
  if (!frame) return;

  const motionBlobs = motionEnabled ? motionDetector.update(frame) : [];

  // Stop drawing YOLO boxes that are no longer being refreshed (slow model,
  // failed run) instead of leaving them frozen on screen.
  if (yoloDets.length && now - yoloTime > Math.max(YOLO_MIN_TTL_MS, detector.lastInferMs * 2.5)) {
    yoloDets = [];
  }
  latestDetections = tracker.update(mergeDetections(yoloDets, motionBlobs), motionEnabled ? motionDetector.shift : null, maskRegions(yoloDets));

  const elapsed = performance.now() - now;
  motionMs = motionMs === null ? elapsed : motionMs * 0.9 + elapsed * 0.1;

  if (++framesSinceYolo >= YOLO_EVERY_N && detector.ready && !detector.busy) {
    framesSinceYolo = 0;
    runYolo();
  }

  if (link?.connected && link.role === 'source' && now - lastShare >= SHARE_INTERVAL_MS &&
      link.bufferedAmount < SHARE_MAX_BUFFERED) {
    lastShare = now;
    shareFrame();
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
let fps = null;
let fpsWindowStart = performance.now();

function updateStats() {
  const now = performance.now();
  if (now - fpsWindowStart >= 1000) {
    fps = Math.round(framesInWindow * 1000 / (now - fpsWindowStart));
    framesInWindow = 0;
    fpsWindowStart = now;
  }
  // On a dashboard the tiles add up every connected device.
  const shown = dashboardMode ? dashboard.allDetections() : latestDetections;
  renderer.updateStats({
    fps: dashboardMode ? null : fps,
    count: shown.length,
    backend: backendLabel,
    inferMs: detector.lastInferMs,
    motionMs,
    resolution: procW ? procW + '×' + procH : '',
    classCounts: countByClass(shown),
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

// --- Linking devices ---
const remoteSources = new Map();   // sourceId -> name

/** "Phone K7": a short tag keeps two phones apart on the dashboard. */
function getDeviceName() {
  const kind = IS_MOBILE ? 'Phone' : 'Computer';
  let tag = '';
  try { tag = localStorage.getItem('bugspotter-device-tag') || ''; } catch { /* storage unavailable */ }
  if (!/^[A-Z0-9]{2}$/.test(tag)) {
    tag = newRoomCode().slice(0, 2);
    try { localStorage.setItem('bugspotter-device-tag', tag); } catch { /* keep it for this visit only */ }
  }
  return kind + ' ' + tag;
}
const deviceName = getDeviceName();

function renderDeviceList() {
  deviceList.textContent = '';
  for (const name of remoteSources.values()) {
    const item = document.createElement('div');
    item.className = 'device-item';
    const dot = document.createElement('span');
    dot.className = 'device-dot connected';
    item.append(dot, ' ' + name);
    deviceList.appendChild(item);
  }
}

/** Swap the local video for the grid of remote feeds, or back. */
function setDashboardMode(on) {
  if (dashboardMode === on) return;
  dashboardMode = on;
  document.body.classList.toggle('dashboard-mode', on);
  dashboardEl.hidden = !on;
  if (on) {
    video.pause();              // this device only watches; no need to analyse its own video
    resetPipeline();
    hideStatus();
  } else {
    dashboard.clear();
    if (video.currentSrc || video.srcObject) video.play().catch(() => {});
  }
}

function joinLink() {
  return location.origin + location.pathname + '?room=' + peerLink.room + (peerLink.broker ? '&peer=' + peerLink.broker : '');
}

function updateLinkUi() {
  const state = link ? link.state : 'disconnected';
  const viaRoom = link === peerLink;
  const role = link?.role;
  const busy = state !== 'disconnected';

  connectionStatus.className = 'conn-status ' + (state === 'connected' ? 'connected' : 'disconnected');
  connectionStatus.textContent = {
    disconnected: 'Not connected',
    connecting: 'Connecting…',
    connected: role === 'viewer' ? 'Dashboard open' : 'Sharing this device',
    reconnecting: link?.detail ? link.detail + ' — retrying…' : 'Reconnecting…',
  }[state];

  joinBtn.textContent = busy && viaRoom && role === 'source' ? 'Stop sharing' : 'Share this device';
  dashboardBtn.textContent = busy && viaRoom && role === 'viewer' ? 'Close dashboard' : 'Open a dashboard';
  connectBtn.textContent = busy && !viaRoom ? 'Disconnect' : 'Connect';
  joinBtn.disabled = busy && !(viaRoom && role === 'source');
  dashboardBtn.disabled = busy && !(viaRoom && role === 'viewer');
  connectBtn.disabled = busy && viaRoom;
  roomInput.disabled = busy;
  serverUrl.disabled = busy;

  const hosting = busy && viaRoom && role === 'viewer';
  roomInfo.hidden = !hosting;
  if (hosting) roomCodeEl.textContent = peerLink.room;
}

let lastLoggedState = '';
function onLinkState(state) {
  if (!link) return;
  const viewing = link.role === 'viewer';
  if (state === 'connected' && viewing) {
    setDashboardMode(true);
    dashboard.setEmptyMessage(link === peerLink
      ? 'No devices yet. On each phone open this page, enter room code ' + peerLink.room + ' and tap "Share this device".'
      : 'Waiting for devices to connect to the relay…');
  }
  if (state === 'disconnected') {
    setDashboardMode(false);
    remoteSources.clear();
    renderDeviceList();
  }

  const message = {
    connecting: '',
    connected: viewing ? (link === peerLink ? 'Dashboard open, room ' + peerLink.room : 'Connected to relay as viewer')
      : (link === peerLink ? 'Sharing with room ' + peerLink.room : 'Connected to relay as source'),
    reconnecting: link.detail || 'Connection lost, retrying…',
    disconnected: link.detail || 'Disconnected',
  }[state];
  if (message && message !== lastLoggedState) log(message);
  lastLoggedState = message;
  updateLinkUi();
}

const linkActive = () => !!link && link.state !== 'disconnected';

function startLink(which, target, role) {
  link = which;
  lastLoggedState = '';
  try {
    which.connect(target, role, deviceName);
  } catch (e) {
    link = null;
    log(e.message);
    updateLinkUi();
    return false;
  }
  return true;
}

function stopLink() {
  if (link) link.disconnect();
  link = null;
  updateLinkUi();
}

function setupLinks() {
  for (const l of [peerLink, relayLink]) {
    l.onStateChange = (state) => { if (l === link) onLinkState(state); };
    l.onSources = (sources) => {
      if (l !== link) return;
      remoteSources.clear();
      for (const s of sources) remoteSources.set(s.id, s.name);
      renderDeviceList();
      dashboard.setSources(sources);
    };
    l.onFrame = (id, name, jpeg, detections, width, height) => {
      if (l === link) dashboard.showFrame(id, name, jpeg, detections, width, height);
    };
    l.onDetections = (id, name, detections, width, height) => {
      if (l === link) dashboard.showFrame(id, name, null, detections, width, height);
    };
  }

  // Room codes
  joinBtn.addEventListener('click', () => {
    if (linkActive()) { stopLink(); return; }
    const code = normalizeRoomCode(roomInput.value);
    if (!code) {
      log('Enter the 6-character room code shown on the dashboard.');
      roomInput.focus();
      return;
    }
    roomInput.value = code;
    startLink(peerLink, code, 'source');
  });
  roomInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') joinBtn.click(); });

  dashboardBtn.addEventListener('click', () => {
    if (linkActive()) { stopLink(); return; }
    startLink(peerLink, newRoomCode(), 'viewer');
  });

  copyLinkBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(joinLink());
      log('Phone link copied. Opening it on a phone joins this dashboard.');
    } catch {
      log('Phone link: ' + joinLink());
    }
  });

  // Relay server (server.js). When the page is served by it, it is on the same host.
  if (location.protocol === 'http:' && location.port) serverUrl.value = 'ws://' + location.host;
  const relayRole = params.get('role') === 'viewer' ? 'viewer' : 'source';
  connectBtn.addEventListener('click', () => {
    if (linkActive()) { stopLink(); return; }
    const url = serverUrl.value.trim() || 'ws://localhost:8080';
    if (!startLink(relayLink, url, relayRole)) return;
    if (location.protocol === 'https:' && url.startsWith('ws://') && !/^ws:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(url)) {
      log('Browsers block ws:// from an https:// page — use wss://, or use a room code instead.');
    }
  });

  updateLinkUi();
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
  if (e.code === 'KeyC' && !dashboardMode) captureFrame();
});

// Browsers that refuse autoplay need a gesture; any tap on the video will do.
view.addEventListener('click', () => {
  if (video.paused && (video.currentSrc || video.srcObject)) {
    video.play().then(hideStatus).catch(() => {});
  }
});

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
  setupLinks();

  const invitedRoom = normalizeRoomCode(params.get('room'));
  if (invitedRoom) {
    // Opened from a dashboard's phone link: use the camera and join right away.
    roomInput.value = invitedRoom;
    sourceSelect.value = 'camera';
    startLink(peerLink, invitedRoom, 'source');
  } else if (!SAMPLES[sourceSelect.value]) {
    // Otherwise start on a sample; never open the camera or a file dialog unprompted.
    sourceSelect.value = DEFAULT_SOURCE;
  }
  uploadArea.classList.remove('visible');
  setSource(sourceSelect.value);

  scheduleFrame();
  setInterval(updateStats, STATS_INTERVAL_MS);
  loadModel();   // in the background: video and motion detection start right away
}

init();
