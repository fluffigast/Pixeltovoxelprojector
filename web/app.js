// app.js — Main application: video input, detection loop, UI controls, multi-phone networking

import { Detector } from './detector.js';
import { MotionDetector } from './motion.js';
import { Tracker } from './tracker.js';
import { Renderer, setDwellNeeded } from './renderer.js';
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

const IS_MOBILE = deviceKind() === 'Phone';   // UA Client Hints / pointer type, see deviceKind()
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
const dwellInput = $('dwellInput');
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
  if (frameCanvas) startCaptureCheck();
  analysedMediaTime = -1;
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
// Some Android GPUs hand a hardware-decoded video to a CPU-readable canvas as
// black or as the same picture over and over; the page then plays fine but
// the detector sees nothing. CaptureCheck notices that from a few sampled
// pixels and getFrame() falls back to the next way of reading the frame.
const CAPTURE_MODES = [
  { name: 'offscreen canvas', make: (w, h) => createCanvas(w, h), opts: { willReadFrequently: true } },
  { name: 'GPU canvas', make: (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }, opts: {} },
  { name: 'VideoFrame', make: (w, h) => createCanvas(w, h), opts: { willReadFrequently: true }, videoFrame: true },
  { name: 'WebGL', make: (w, h) => createCanvas(w, h), opts: { willReadFrequently: true }, webgl: true },
  // Firefox for Android reads a hardware-decoded file as black everywhere
  // above, but frames of a MediaStream (the camera path) read fine; so play
  // the file into a stream and read that.
  { name: 'captured stream', make: (w, h) => createCanvas(w, h), opts: { willReadFrequently: true }, stream: true },
];

let capVideo = null, capSource = '';
const lastStreamSig = new Uint8Array(64);
function sameAsLastStreamFrame(frame) {
  const { data, width: w, height: h } = frame;
  let same = true;
  for (let k = 0; k < 64; k++) {
    const i = (Math.floor(((k >> 3) + 0.5) * h / 8) * w + Math.floor(((k & 7) + 0.5) * w / 8)) * 4 + 1;
    if (data[i] !== lastStreamSig[k]) { same = false; lastStreamSig[k] = data[i]; }
  }
  return same;
}
/** A hidden <video> showing video.captureStream(), rebuilt when the source changes. */
function capturedVideo() {
  // A new source (even the same file again) ends the old stream's tracks.
  const src = sourceToken + ' ' + (video.currentSrc || 'stream');
  const track = capVideo?.srcObject?.getVideoTracks?.()[0];
  if (capVideo && capSource === src && track && track.readyState !== 'ended') return capVideo;
  const grab = video.captureStream || video.mozCaptureStream;
  if (!grab) throw new Error('captureStream unsupported');
  if (!capVideo) {
    // In the document (invisible) so the browser keeps decoding it at full rate.
    capVideo = Object.assign(document.createElement('video'), { muted: true, playsInline: true });
    capVideo.setAttribute('aria-hidden', 'true');
    capVideo.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none';
    document.body.appendChild(capVideo);
  }
  capVideo.srcObject = grab.call(video);
  capVideo.play().catch(() => {});
  capSource = src;
  return capVideo;
}

/** Reads video frames through a WebGL texture: a different GPU path than drawImage. */
class GlGrabber {
  constructor() {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl', { preserveDrawingBuffer: true, premultipliedAlpha: false, antialias: false });
    if (!gl) throw new Error('WebGL unavailable');
    const sh = (type, src) => { const o = gl.createShader(type); gl.shaderSource(o, src); gl.compileShader(o); return o; };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, 'attribute vec2 p;varying vec2 t;void main(){t=vec2(p.x*.5+.5,.5-p.y*.5);gl_Position=vec4(p,0.,1.);}'));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, 'precision mediump float;varying vec2 t;uniform sampler2D s;void main(){gl_FragColor=texture2D(s,t);}'));
    gl.linkProgram(prog);
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    this.c = c; this.gl = gl; this.buf = null;
  }

  /** Draws the current video frame into ctx (w x h). */
  grab(src, ctx, w, h) {
    const gl = this.gl, c = this.c;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; this.buf = new ImageData(w, h); }
    gl.viewport(0, 0, w, h);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // readPixels rows run bottom-up; the shader already flipped the picture, so they come out top-down.
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(this.buf.data.buffer));
    ctx.putImageData(this.buf, 0, 0);
  }
}
let glGrabber = null;
const CHECK_FRAMES = 30;          // frames with advancing video time before judging
let captureMode = Math.min(CAPTURE_MODES.length - 1, Math.max(0, parseInt(new URLSearchParams(location.search).get('capture'), 10) || 0));
let captureCheck = null;          // { n, dark, still, prev: Uint8Array, lastTime } while judging; null when settled

function startCaptureCheck() {
  captureCheck = { n: 0, dark: 0, still: 0, prev: null, lastTime: -1 };
}

/** Sample 256 grey values on a grid and judge the capture after CHECK_FRAMES. */
function checkCapture(frame) {
  const c = captureCheck;
  if (!c || video.currentTime === c.lastTime) return;
  c.lastTime = video.currentTime;
  const { data, width: w, height: h } = frame;
  const cur = new Uint8Array(256);
  let sum = 0, diff = 0;
  for (let k = 0; k < 256; k++) {
    const x = Math.floor(((k % 16) + 0.5) * w / 16), y = Math.floor(((k >> 4) + 0.5) * h / 16);
    const i = (y * w + x) * 4;
    const g = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    cur[k] = g; sum += g;
    if (c.prev) diff += Math.abs(g - c.prev[k]);
  }
  if (sum / 256 < 3) c.dark++;
  if (c.prev && diff === 0) c.still++;
  c.prev = cur;
  if (++c.n < CHECK_FRAMES) return;
  const mode = CAPTURE_MODES[captureMode].name;
  const bad = c.dark > CHECK_FRAMES / 2 ? 'black' : c.still > CHECK_FRAMES / 2 ? 'frozen' : '';
  captureCheck = null;
  if (!bad) {
    log('Frame check OK (' + mode + ', mean grey ' + Math.round(sum / 256) + ')');
    return;
  }
  if (!canvasReadbackWorks()) {
    log('This browser blocks pages from reading canvas pixels (privacy / anti-fingerprinting protection), so nothing can be analysed. ' +
      'Allow fingerprinting or canvas access for this site in the browser settings (e.g. Brave Shields), or open the page in Chrome.');
    showStatus('The browser blocks reading video pixels. See the activity log.');
    return;
  }
  if (captureMode + 1 < CAPTURE_MODES.length) {
    captureMode++;
    log('Video frames read as ' + bad + ' via ' + mode + '; switching to ' + CAPTURE_MODES[captureMode].name);
    frameCanvas = null;          // rebuilt by getFrame() with the new mode
  } else {
    log('Video frames read as ' + bad + ' with every capture method; this browser cannot analyse this video. ' +
      'Open the page in Chrome, or try the Camera source.');
    showStatus((/Firefox\//.test(navigator.userAgent) ? 'Firefox' : 'This browser') +
      ' hands the video to the page as a ' + bad + ' picture, so nothing can be detected. Open the page in Chrome, or try the Camera.');
  }
}

/** Draw a known pattern and read it back: privacy protections return zeros or noise instead. */
function canvasReadbackWorks() {
  try {
    const c = document.createElement('canvas');
    c.width = 4; c.height = 4;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = 'rgb(200, 100, 50)';
    ctx.fillRect(0, 0, 4, 4);
    const d = ctx.getImageData(0, 0, 4, 4).data;
    for (let i = 0; i < d.length; i += 4) {
      if (Math.abs(d[i] - 200) > 3 || Math.abs(d[i + 1] - 100) > 3 || Math.abs(d[i + 2] - 50) > 3) return false;
    }
    return true;
  } catch {
    return false;
  }
}

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
  const mode = CAPTURE_MODES[captureMode];
  if (!frameCanvas || procW !== w || procH !== h) {
    procW = w;
    procH = h;
    frameCanvas = mode.make(w, h);
    frameCtx = frameCanvas.getContext('2d', mode.opts);
    resetPipeline();
    updateLayout();
    startCaptureCheck();
  }
  try {
    if (mode.stream) {
      const cv = capturedVideo();
      if (cv.readyState < 2) return null;
      frameCtx.drawImage(cv, 0, 0, w, h);
      // The stream updates on its own clock: a repeated picture is not a new frame.
      const probe = frameCtx.getImageData(0, 0, w, h);
      if (sameAsLastStreamFrame(probe)) return null;
      checkCapture(probe);
      return probe;
    } else if (mode.webgl) {
      glGrabber ||= new GlGrabber();
      glGrabber.grab(video, frameCtx, w, h);
    } else if (mode.videoFrame) {
      const vf = new VideoFrame(video);
      try { frameCtx.drawImage(vf, 0, 0, w, h); } finally { vf.close(); }
    } else {
      frameCtx.drawImage(video, 0, 0, w, h);
    }
  } catch (e) {
    // This way of reading frames does not work here at all: try the next one.
    const failed = mode.name;
    if (captureMode + 1 < CAPTURE_MODES.length) {
      captureMode++;
      frameCanvas = null;
      log('Reading frames via ' + failed + ' failed (' + (e?.message || e) + '); switching to ' + CAPTURE_MODES[captureMode].name);
    } else if (captureCheck) {
      captureCheck = null;
      log('Reading frames via ' + failed + ' failed (' + (e?.message || e) + '); no capture method left.');
    }
    return null;
  }
  const frame = frameCtx.getImageData(0, 0, w, h);
  checkCapture(frame);
  return frame;
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

// The marks belong to the frame that was analysed; by the time they are on
// screen the video has moved on (analysis takes about a frame on a phone),
// and a fly doing 10-25 px per frame would be marked where it was. So each
// mark is drawn ahead along its track velocity by the video time elapsed
// since its frame, in units of the time between analysed frames.
const MAX_LEAD = 3;            // analysed-frame intervals; beyond that a prediction is a guess
let analysedMediaTime = -1;    // video time of the frame latestDetections came from
let mediaStep = 1 / 30;        // smoothed video time between analysed frames

function noteAnalysedFrame(mediaTime) {
  if (analysedMediaTime >= 0) {
    const d = mediaTime - analysedMediaTime;
    if (d > 0 && d < 0.5) mediaStep += (d - mediaStep) * 0.2;
  }
  analysedMediaTime = mediaTime;
}

function render() {
  if (!procW || !layout.w) return;
  const sx = layout.w / procW;
  const sy = layout.h / procH;
  let lead = 0;
  if (analysedMediaTime >= 0 && !video.paused) {
    lead = (video.currentTime - analysedMediaTime) / mediaStep;
    lead = lead > MAX_LEAD ? MAX_LEAD : lead > 0 ? lead : 0;
  }
  renderer.draw(latestDetections.map(d => {
    const ox = (d.vx || 0) * lead, oy = (d.vy || 0) * lead;
    const trail = d.trail && d.trail.map(p => [p[0] * sx, p[1] * sy]);
    if (trail && (ox || oy)) trail.push([(d.x + d.w / 2 + ox) * sx, (d.y + d.h / 2 + oy) * sy]);
    return {
      ...d,
      x: Math.round((d.x + ox) * sx),
      y: Math.round((d.y + oy) * sy),
      w: Math.round(d.w * sx),
      h: Math.round(d.h * sy),
      aimError: d.aimError == null ? null : d.aimError * sx,
      trail,
    };
  }));
}

// Redraw on every display frame so the predicted marks move smoothly between analysed frames.
function renderLoop() {
  if (!dashboardMode && !video.paused) render();
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);

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
  const detections = latestDetections.map(({ id, x, y, w: dw, h: dh, label, confidence, missed, aimError, onTarget, trail }) => ({
    id, x, y, w: dw, h: dh, label, confidence, missed, aimError, onTarget,
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
// requestVideoFrameCallback is preferred; some Android browsers never call it
// back (or stop after a source change), so a watchdog falls back to
// requestAnimationFrame plus a currentTime check when a playing video has
// produced no analysed frame for a second.
let useVideoFrameCallback = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
let lastMediaTime = -1;
let lastProcessed = performance.now();
let lastErrorLogged = '';

function onFrame(now, metadata) {
  try {
    if (useVideoFrameCallback || video.currentTime !== lastMediaTime) {
      lastMediaTime = video.currentTime;
      lastProcessed = performance.now();
      noteAnalysedFrame(useVideoFrameCallback && metadata ? metadata.mediaTime : video.currentTime);
      processFrame();
    }
  } catch (e) {
    // Keep going, but say what broke: without this a failing frame loop only shows as "0 FPS".
    const msg = String(e?.message || e);
    if (msg !== lastErrorLogged) { lastErrorLogged = msg; log('Analysis error: ' + msg); console.error(e); }
  } finally {
    scheduleFrame();
  }
}

function scheduleFrame() {
  if (useVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

setInterval(() => {
  const playing = !video.paused && !video.ended && video.readyState >= 2;
  if (!useVideoFrameCallback || !playing || dashboardMode) { if (!playing) lastProcessed = performance.now(); return; }
  if (performance.now() - lastProcessed > 1000) {
    useVideoFrameCallback = false;
    log('Video frame callbacks stopped; analysing on animation frames instead');
    lastProcessed = performance.now();
    requestAnimationFrame(onFrame);
  }
}, 500);

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
  const kind = deviceKind();
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
    const item = document.createElement('li');
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
    log(e.message + (e.cause ? ' (' + e.cause.message + ')' : ''));
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
  // Connecting only ever starts from this click, so a Local Network Access
  // prompt is tied to a user gesture and never fires on page load.
  connectBtn.addEventListener('click', async () => {
    if (linkActive()) { stopLink(); return; }
    const url = serverUrl.value.trim() || 'ws://localhost:8080';
    const space = addressSpaceFor(url);
    const state = await lnaPermission(space);
    showLnaHelp(space, state);
    if (state === 'denied') { log('Local network access denied for ' + url); return; }
    if (linkActive()) return;   // something else connected while the permission was checked
    if (!startLink(relayLink, url, relayRole)) return;
    if (location.protocol === 'https:' && url.startsWith('ws://') && !space) {
      log('Browsers block ws:// to a public host from an https:// page — use wss://, or use a room code instead.');
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
function applyDwellNeeded() {
  setDwellNeeded(dwellInput.value);
}
dwellInput.addEventListener('input', applyDwellNeeded);

motionToggle.addEventListener('change', applyMotionSettings);
motionSlider.addEventListener('input', applyMotionSettings);

captureBtn.addEventListener('click', captureFrame);
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;   // leave Ctrl+C etc. alone
  if (e.target.matches?.('input, select, textarea')) return;
  if (e.code === 'KeyC' && !dashboardMode) captureFrame();
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

// --- Staying up to date ---
// The host lets the browser reuse the page for ten minutes without asking, so a
// visit shortly after a deploy can show the previous version. This page's
// version is the stamp in its own URL (written by tools/stamp_web.mjs). If the
// server already has another one, reload once to pick it up.
const APP_VERSION = new URL(import.meta.url).searchParams.get('v');

async function checkForUpdate() {
  if (!APP_VERSION) return;   // unstamped copy (local development)
  let latest = null;
  try {
    const res = await fetch(location.pathname, { cache: 'no-store' });
    if (res.ok) latest = /<script type="module" src="app\.js\?v=([0-9a-f]+)">/.exec(await res.text())?.[1] ?? null;
  } catch { /* offline: nothing to compare with */ }
  if (!latest || latest === APP_VERSION) return;

  const KEY = 'bugspotter-reloaded-for';
  let tried = null;
  try { tried = sessionStorage.getItem(KEY); } catch { /* storage unavailable */ }
  if (tried === latest) {
    // Already reloaded for this version and still got the old page.
    log('A newer version of this page is published, but the browser keeps showing the old one. Reload without cache to update.');
    return;
  }
  try { sessionStorage.setItem(KEY, latest); } catch { return; }   // no way to remember: do not risk a reload loop
  location.reload();
}

function init() {
  checkForUpdate();
  applyConfidence();
  applyMotionSettings();
  applyDwellNeeded();
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
  const brands = navigator.userAgentData?.brands?.filter(b => !/Not.?A.?Brand/i.test(b.brand)).map(b => b.brand + ' ' + b.version).join(', ');
  log('Browser: ' + (brands || navigator.userAgent.replace(/^Mozilla\/5\.0 /, '')) + (navigator.userAgentData?.platform ? ' on ' + navigator.userAgentData.platform : ''));
}

init();
