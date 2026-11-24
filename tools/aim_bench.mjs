// aim_bench.mjs — How often is the aim on the fly? Scored against ground truth.
//
// Flies are painted into real sample clips at known positions. At every frame
// each reported track has an aim point for the NEXT frame (centre + velocity,
// exactly what tracker.js judges itself by). The aim is "on" if it lands inside
// the body of the painted fly the track was sitting on. The score is the share
// of ALL fly-frames with the aim on, so losing the fly counts against it.
//
//   steady   cruising insect, gentle curves, 3-12 px/frame
//   erratic  darting insect, sharp turns every frame, 6-24 px/frame
//   slow     walking on glass, 0.4-2 px/frame
//
// For scale: with perfect positions and last-step velocity the ceiling is about
// 99% (steady), 32% (erratic) and 100% (slow). No one-frame predictor can do
// much better on "erratic"; that needs a faster camera, not a better tracker.
//
// Usage: node tools/aim_bench.mjs [clip ...]      TRACKER='{...}' for tracker options
// Needs ffmpeg and ffprobe on the PATH.

import { spawnSync } from 'node:child_process';
import { openSync, readSync, closeSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MotionDetector } from '../web/motion.js';
import { Tracker } from '../web/tracker.js';

const SAMPLES = fileURLToPath(new URL('../web/samples/', import.meta.url));
const trackerOptions = JSON.parse(process.env.TRACKER || '{}');
const FRAMES = 240;
const MAX_SIDE = 1280;

function eachFrame(name, cb) {
  const file = join(SAMPLES, name + '.mp4');
  const probe = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', file]);
  if (probe.status !== 0) throw new Error('ffprobe failed for ' + file);
  let [w, h] = probe.stdout.toString().trim().split(',').map(Number);
  const s = Math.min(1, MAX_SIDE / Math.max(w, h)); w = Math.round(w * s); h = Math.round(h * s);
  const dir = mkdtempSync(join(tmpdir(), 'aimbench-'));
  const raw = join(dir, 'frames.rgba');
  try {
    spawnSync('ffmpeg', ['-v', 'error', '-y', '-i', file, '-vf', `scale=${w}:${h}`, '-frames:v', String(FRAMES), '-f', 'rawvideo', '-pix_fmt', 'rgba', raw]);
    const fd = openSync(raw, 'r'), size = w * h * 4;
    for (let i = 0; ; i++) { const data = new Uint8ClampedArray(size); if (readSync(fd, data, 0, size, null) < size) break; cb({ data, width: w, height: h }, i); }
    closeSync(fd);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const KINDS = {
  steady:  { speed: [3, 12], turn: 0.12, jump: 0.01, blur: true },   // cruising: gentle curves
  erratic: { speed: [6, 24], turn: 0.9,  jump: 0.10, blur: true },   // darting: sharp, frequent turns
  slow:    { speed: [0.4, 2], turn: 0.25, jump: 0.02, blur: false }, // walking on glass
};
function painter(kind, count, seed) {
  const K = KINDS[kind], R = rng(seed); let flies = null;
  const lum = (f, x, y) => { const i = ((y | 0) * f.width + (x | 0)) * 4; return (f.data[i] * 77 + f.data[i + 1] * 150 + f.data[i + 2] * 29) >> 8; };
  const stamp = (f, cx, cy, r, a0) => { for (let y = Math.max(0, Math.floor(cy - r - 1)); y <= Math.min(f.height - 1, Math.ceil(cy + r + 1)); y++) for (let x = Math.max(0, Math.floor(cx - r - 1)); x <= Math.min(f.width - 1, Math.ceil(cx + r + 1)); x++) { const a = a0 * Math.min(1, Math.max(0, r + 0.5 - Math.hypot(x + 0.5 - cx, y + 0.5 - cy))); if (a <= 0) continue; const i = (y * f.width + x) * 4; for (let c = 0; c < 3; c++) f.data[i + c] = f.data[i + c] * (1 - a) + 25 * a; } };
  return (f) => {
    const spawn = (id) => ({ id, x: 40 + R() * (f.width - 80), y: 40 + R() * (f.height - 80), dir: R() * 6.283, speed: K.speed[0] + R() * (K.speed[1] - K.speed[0]), r: 2 + R() * 2.5, alpha: 0.6 + R() * 0.35 });
    if (!flies) flies = Array.from({ length: count }, (_, i) => spawn(i));
    const out = [];
    for (let k = 0; k < flies.length; k++) {
      const fly = flies[k];
      fly.dir += (R() - 0.5) * 2 * K.turn; if (R() < K.jump) fly.dir += (R() - 0.5) * 4;
      fly.speed = Math.min(K.speed[1], Math.max(K.speed[0], fly.speed + (R() - 0.5) * 0.15 * (K.speed[1] - K.speed[0])));
      const px = fly.x, py = fly.y; fly.x += Math.cos(fly.dir) * fly.speed; fly.y += Math.sin(fly.dir) * fly.speed;
      if (fly.x < 12 || fly.y < 12 || fly.x > f.width - 12 || fly.y > f.height - 12) { flies[k] = spawn(fly.id + 1000); continue; }
      const steps = K.blur ? Math.max(1, Math.ceil(fly.speed * 0.5 / fly.r)) : 1;
      for (let s = 0; s < steps; s++) { const t = 1 - 0.5 * s / steps; stamp(f, px + (fly.x - px) * t, py + (fly.y - py) * t, fly.r, fly.alpha / Math.sqrt(steps)); }
      out.push({ id: fly.id, x: fly.x, y: fly.y, r: fly.r, speed: fly.speed, visible: lum(f, fly.x + 8, fly.y + 8) > 90 });
    }
    return out;
  };
}
const clips = process.argv.slice(2).length ? process.argv.slice(2) : ['new_video_fixed', 'insects_flying_forest', 'new_video'];
let sum = 0;
console.log('kind      aim on the fly   tracked with an aim   on when aimed   miss p50 / p90 (px)');
for (const kind of Object.keys(KINDS)) {
  let flyFrames = 0, aimed = 0, on = 0; const errs = [];
  for (const clip of clips) {
    const motion = new MotionDetector(), tracker = new Tracker(trackerOptions), paint = painter(kind, 4, 21);
    let prevAims = [];   // aims made last frame, each tied to the fly its track sat on
    eachFrame(clip, (f, i) => {
      const gt = paint(f);
      const tracks = tracker.update(motion.update(f), motion.shift, []);
      if (i >= 10) for (const g of gt) {
        if (!g.visible) continue;
        flyFrames++;
        const mine = prevAims.filter(a => a.flyId === g.id);
        if (!mine.length) continue;
        aimed++;
        const e = Math.min(...mine.map(a => Math.hypot(a.ax - g.x, a.ay - g.y)));
        errs.push(e);
        if (e <= g.r + 1) on++;
      }
      prevAims = [];
      for (const d of tracks) {
        const cx = d.x + d.w / 2, cy = d.y + d.h / 2;
        let best = null, bd = Infinity;
        for (const g of gt) { const dist = Math.hypot(cx - g.x, cy - g.y); if (dist < g.r + 6 + g.speed * 0.5 && dist < bd) { bd = dist; best = g; } }
        if (best) prevAims.push({ flyId: best.id, ax: d.aimX, ay: d.aimY });
      }
    });
  }
  errs.sort((a, b) => a - b);
  sum += on / flyFrames;
  const pct = (v) => (100 * v).toFixed(1).padStart(5) + '%';
  console.log(`${kind.padEnd(10)}${pct(on / flyFrames).padStart(12)}${pct(aimed / flyFrames).padStart(20)}${pct(on / (aimed || 1)).padStart(18)}${((errs[errs.length >> 1] ?? 0).toFixed(1) + ' / ' + (errs[Math.floor(errs.length * 0.9)] ?? 0).toFixed(1)).padStart(18)}`);
}
console.log('mean      ' + ((100 * sum / 3).toFixed(1) + '%').padStart(12));
