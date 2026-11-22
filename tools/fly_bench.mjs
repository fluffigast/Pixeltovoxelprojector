// fly_bench.mjs — How well does web/motion.js + web/tracker.js find flies?
//
// There is no labelled fly footage in this repo, so this measures two things
// on the real sample clips:
//   recall  — synthetic flies (dark, 2.5–7 px, 4–28 px/frame, erratic, motion
//             blurred) are painted into the frames at known positions; recall
//             is the share of visible fly-frames that carry a mark.
//   marks   — marks per frame on the untouched clip. These are false alarms
//             plus whatever really flies in the clip, so compare runs with
//             each other rather than reading the number as an error rate.
//
// Usage:  node tools/fly_bench.mjs [clip ...]
//         SKIP=2 node tools/fly_bench.mjs            # every 2nd frame (slow device)
//         MOTION='{"threshold":9}' TRACKER='{"minHits":3}' node tools/fly_bench.mjs
// Needs ffmpeg and ffprobe on the PATH.

import { spawnSync } from 'node:child_process';
import { openSync, readSync, closeSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MotionDetector } from '../web/motion.js';
import { Tracker } from '../web/tracker.js';

const SAMPLES = fileURLToPath(new URL('../web/samples/', import.meta.url));
const CLIPS = ['new_video_fixed', 'new_video', 'new_video2', 'drone_crossing_sky', 'insects_flying_forest', 'bees_hive'];
const MAX_FRAMES = 300;
const MAX_SIDE = 1280;
const FLIES = 5;
const SKIP = Number(process.env.SKIP || 1);
const motionOptions = JSON.parse(process.env.MOTION || '{}');
const trackerOptions = JSON.parse(process.env.TRACKER || '{}');

/** Decode a clip to raw RGBA once and call cb(frame) for each frame. */
function eachFrame(name, cb) {
  const file = join(SAMPLES, name + '.mp4');
  const probe = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', file]);
  if (probe.status !== 0) throw new Error('ffprobe failed for ' + file);
  let [w, h] = probe.stdout.toString().trim().split(',').map(Number);
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  w = Math.round(w * scale); h = Math.round(h * scale);
  const dir = mkdtempSync(join(tmpdir(), 'flybench-'));
  const raw = join(dir, 'frames.rgba');
  try {
    spawnSync('ffmpeg', ['-v', 'error', '-y', '-i', file, '-vf', `scale=${w}:${h}`, '-frames:v', String(MAX_FRAMES), '-f', 'rawvideo', '-pix_fmt', 'rgba', raw]);
    const fd = openSync(raw, 'r');
    const size = w * h * 4;
    for (let i = 0; ; i++) {
      const data = new Uint8ClampedArray(size);
      if (readSync(fd, data, 0, size, null) < size) break;
      if (i % SKIP === 0) cb({ data, width: w, height: h });
    }
    closeSync(fd);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** Small deterministic random generator, so every run paints the same flies. */
function random(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Returns paint(frame) → ground truth [{x, y, r, speed, visible}] for that frame. */
function flyPainter(count, seed) {
  const R = random(seed);
  let flies = null;
  const luminance = (f, x, y) => { const i = ((y | 0) * f.width + (x | 0)) * 4; return (f.data[i] * 77 + f.data[i + 1] * 150 + f.data[i + 2] * 29) >> 8; };
  const stamp = (f, cx, cy, r, alpha) => {            // darken a soft disc
    for (let y = Math.max(0, Math.floor(cy - r - 1)); y <= Math.min(f.height - 1, Math.ceil(cy + r + 1)); y++) {
      for (let x = Math.max(0, Math.floor(cx - r - 1)); x <= Math.min(f.width - 1, Math.ceil(cx + r + 1)); x++) {
        const a = alpha * Math.min(1, Math.max(0, r + 0.5 - Math.hypot(x + 0.5 - cx, y + 0.5 - cy)));
        if (a <= 0) continue;
        const i = (y * f.width + x) * 4;
        for (let c = 0; c < 3; c++) f.data[i + c] = f.data[i + c] * (1 - a) + 25 * a;
      }
    }
  };
  return (f) => {
    const spawn = () => ({ x: 40 + R() * (f.width - 80), y: 40 + R() * (f.height - 80), dir: R() * 6.283, speed: 4 + R() * 24, r: 1.25 + R() * 2.25, alpha: 0.5 + R() * 0.45 });
    if (!flies) flies = Array.from({ length: count }, spawn);
    const truth = [];
    for (let k = 0; k < flies.length; k++) {
      const fly = flies[k];
      fly.dir += (R() - 0.5) * 1.2;
      if (R() < 0.08) fly.dir += (R() - 0.5) * 4;       // sudden turn
      fly.speed = Math.min(28, Math.max(4, fly.speed + (R() - 0.5) * 6));
      const px = fly.x, py = fly.y;
      fly.x += Math.cos(fly.dir) * fly.speed;
      fly.y += Math.sin(fly.dir) * fly.speed;
      if (fly.x < 10 || fly.y < 10 || fly.x > f.width - 10 || fly.y > f.height - 10) { flies[k] = spawn(); continue; }
      const visible = luminance(f, fly.x, fly.y) > 100;   // a dark fly only shows on a bright background
      const steps = Math.max(1, Math.ceil(fly.speed * 0.5 / Math.max(1, fly.r)));   // blur over half a frame of travel
      for (let s = 0; s < steps; s++) {
        const t = 1 - 0.5 * s / steps;
        stamp(f, px + (fly.x - px) * t, py + (fly.y - py) * t, fly.r, fly.alpha / Math.sqrt(steps));
      }
      truth.push({ x: fly.x, y: fly.y, r: fly.r, speed: fly.speed, visible });
    }
    return truth;
  };
}

function run(name, flyCount) {
  const motion = new MotionDetector(motionOptions);
  const tracker = new Tracker(trackerOptions);
  const paint = flyCount ? flyPainter(flyCount, 7) : null;
  let frames = 0, visible = 0, found = 0, otherMarks = 0, ms = 0;
  eachFrame(name, (f) => {
    const truth = paint ? paint(f) : [];
    const t0 = performance.now();
    const marks = tracker.update(motion.update(f), motion.shift);
    ms += performance.now() - t0;
    if (++frames <= 8) return;                              // let the tracker warm up
    const m = marks.map(d => ({ cx: d.x + d.w / 2, cy: d.y + d.h / 2, half: Math.max(d.w, d.h) / 2, onFly: false }));
    for (const g of truth) {
      const reach = 10 + 2 * g.r + g.speed * 0.5;
      let hit = false;
      for (const d of m) {
        const dist = Math.hypot(d.cx - g.x, d.cy - g.y);
        if (dist < reach + d.half) hit = true;
        if (dist < 30 + d.half) d.onFly = true;
      }
      if (g.visible) { visible++; if (hit) found++; }
    }
    otherMarks += m.filter(d => !d.onFly).length;
  });
  const counted = Math.max(1, frames - 8);
  return { recall: visible ? found / visible : NaN, marks: otherMarks / counted, ms: ms / Math.max(1, frames) };
}

const clips = process.argv.slice(2).length ? process.argv.slice(2) : CLIPS;
let recallSum = 0, marksSum = 0;
console.log('clip                    recall on painted flies   marks/frame on untouched clip   ms/frame');
for (const name of clips) {
  const clean = run(name, 0);
  const withFlies = run(name, FLIES);
  recallSum += withFlies.recall;
  marksSum += clean.marks;
  console.log(`${name.padEnd(24)}${(withFlies.recall * 100).toFixed(0).padStart(6)}%${clean.marks.toFixed(2).padStart(31)}${clean.ms.toFixed(1).padStart(21)}`);
}
console.log(`${'mean'.padEnd(24)}${(recallSum / clips.length * 100).toFixed(0).padStart(6)}%${(marksSum / clips.length).toFixed(2).padStart(31)}`);
