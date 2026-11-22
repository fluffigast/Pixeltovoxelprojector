// bench.mjs — Score web/motion.js + web/tracker.js the way app.js runs them.
//
// Two kinds of evidence per clip:
//   real   — visually verified ground truth in tools/gt/<clip>.json (tracks of
//            real insects / birds / drones, one point per visible frame).
//            Reports recall by kind (fast / slow / big), how quickly a track is
//            picked up, and false positives (output not explained by any
//            listed object).
//   synth  — synthetic dark flies painted into the frames at known positions:
//            "fast" (4–28 px/frame, motion blurred) and "slow" (0.2–2 px/frame
//            with pauses, like a fly walking on glass).
//
// Usage:
//   node tools/bench.mjs [clip ...] [--motion web/motion.js] [--tracker web/tracker.js]
//        [--mopt '{"threshold":20}'] [--topt '{}'] [--skip 2] [--no-synth] [--no-real]
//        [--maxside 960] [--json out.json] [--dump dir] [--gt tools/gt]
//   --skip N     feed every Nth frame only (a device that cannot keep up)
//   --maxside N  downscale frames so the longer side is N px, as app.js does on
//                phones (960); ground truth is scaled to match
//   --dump dir write per-frame outputs of the real run to dir/<clip>.out.json
//
// Frames are decoded once with ffmpeg into BENCH_CACHE (default: the system
// temp dir) as raw RGBA; frame index = decode order, as in the GT files.
// BENCH_GT overrides the ground-truth directory (same as --gt).

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const SAMPLES = path.join(ROOT, 'web', 'samples');
const CACHE = process.env.BENCH_CACHE || path.join(os.tmpdir(), 'bugspotter-bench');
const ALL_CLIPS = ['new_video_fixed', 'new_video', 'new_video2', 'drone_crossing_sky', 'insects_flying_forest', 'bees_hive'];
const SYNTH_CLIPS = ['new_video_fixed', 'new_video', 'new_video2'];
const SYNTH_FRAMES = 300;

const argv = process.argv.slice(2);
const FLAGS = new Set(['--no-synth', '--no-real']);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const has = (k) => argv.includes('--' + k);
const positional = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--') && !FLAGS.has(argv[i - 1])));
const clips = positional.length ? positional : ALL_CLIPS;
const motionPath = path.resolve(opt('motion', path.join(ROOT, 'web', 'motion.js')));
const trackerPath = path.resolve(opt('tracker', path.join(ROOT, 'web', 'tracker.js')));
const motionOptions = JSON.parse(opt('mopt', '{}'));
const trackerOptions = JSON.parse(opt('topt', '{}'));
const SKIP = Number(opt('skip', 1));
const MAXSIDE = Number(opt('maxside', 0));
const gtDir = path.resolve(opt('gt', process.env.BENCH_GT || path.join(ROOT, 'tools', 'gt')));
const dumpDir = opt('dump', null);

const { MotionDetector } = await import(pathToFileURL(motionPath));
const { Tracker } = await import(pathToFileURL(trackerPath));

// ---------------------------------------------------------------- frames
function probe(file) {
  const r = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', file]);
  if (r.status !== 0) throw new Error('ffprobe failed for ' + file);
  const [w, h, rate] = r.stdout.toString().trim().split(',');
  const [num, den] = rate.split('/').map(Number);
  return { w: +w, h: +h, fps: num / (den || 1) };
}

/** Decode (once) to raw RGBA in the cache. `scale` is processed px per native px. */
function clipMeta(name) {
  fs.mkdirSync(CACHE, { recursive: true });
  const file = path.join(SAMPLES, name + '.mp4');
  const native = path.join(CACHE, name + '.meta.json');
  const src = fs.existsSync(native) ? JSON.parse(fs.readFileSync(native, 'utf8')) : probe(file);
  const scale = MAXSIDE && Math.max(src.w, src.h) > MAXSIDE ? MAXSIDE / Math.max(src.w, src.h) : 1;
  const w = Math.round(src.w * scale), h = Math.round(src.h * scale);
  const tag = scale === 1 ? name : name + '@' + MAXSIDE;
  const raw = path.join(CACHE, tag + '.rgba'), metaFile = path.join(CACHE, tag + '.meta.json');
  if (!fs.existsSync(raw) || !fs.existsSync(metaFile)) {
    const filter = scale === 1 ? [] : ['-vf', `scale=${w}:${h}:flags=bilinear`];
    const r = spawnSync('ffmpeg', ['-v', 'error', '-y', '-i', file, ...filter, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgba', raw]);
    if (r.status !== 0) throw new Error('ffmpeg failed for ' + file + ': ' + r.stderr);
    fs.writeFileSync(metaFile, JSON.stringify({ w, h, n: fs.statSync(raw).size / (w * h * 4), fps: src.fps ?? probe(file).fps }));
  }
  const meta = JSON.parse(fs.readFileSync(metaFile, 'utf8'));
  return { fps: 30, ...meta, raw, scale };
}

/** cb(frame, t) for every SKIPth frame in [0, limit). The same buffer is reused. */
function eachFrame(meta, limit, cb) {
  const size = meta.w * meta.h * 4;
  const data = new Uint8ClampedArray(size);
  const fd = fs.openSync(meta.raw, 'r');
  try {
    for (let t = 0; t < Math.min(limit, meta.n); t++) {
      if (t % SKIP) continue;
      fs.readSync(fd, data, 0, size, t * size);
      cb({ data, width: meta.w, height: meta.h }, t);
    }
  } finally { fs.closeSync(fd); }
}

/** The app's per-frame pipeline (see processFrame in web/app.js), without YOLO. */
function makePipeline() {
  const motion = new MotionDetector(motionOptions);
  const tracker = new Tracker(trackerOptions);
  return (frame) => tracker.update(motion.update(frame), motion.shift, []);
}

// ---------------------------------------------------------------- real GT
const HIT_PX = 20, DC_PX = 30, MOVING_PX_PER_FRAME = 0.2;

function scoreReal(name) {
  const gtFile = path.join(gtDir, name + '.json');
  if (!fs.existsSync(gtFile)) return null;
  const gt = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
  const meta = clipMeta(name);
  if (meta.scale !== 1) {                       // ground truth is in native pixels
    const k = meta.scale;
    for (const tr of [...gt.tracks, ...(gt.ignoreTracks || [])]) tr.pts = tr.pts.map(([t, ...v]) => [t, ...v.map(x => x * k)]);
    for (const g of gt.ignoreRegions || []) { g.x *= k; g.y *= k; g.w *= k; g.h *= k; }
  }
  const [f0, f1] = gt.frames || [0, meta.n];
  const step = makePipeline();
  const outs = new Map();   // t -> [{cx, cy, w, h, id, missed}]
  let ms = 0, processed = 0;
  eachFrame(meta, f1, (frame, t) => {
    const t0 = performance.now();
    const tracks = step(frame);
    ms += performance.now() - t0; processed++;
    outs.set(t, tracks.map(d => ({ cx: d.x + d.w / 2, cy: d.y + d.h / 2, w: d.w, h: d.h, id: d.id, missed: d.missed || 0, label: d.label })));
  });
  if (dumpDir) {
    fs.mkdirSync(dumpDir, { recursive: true });
    fs.writeFileSync(path.join(dumpDir, name + '.out.json'), JSON.stringify(Object.fromEntries(outs)));
  }
  // With --skip only analysed frames are scored: in between, the marks on
  // screen are simply one frame old.
  const shownAt = (t) => outs.get(t) || [];

  const kinds = {};
  const bump = (k, hit) => { const o = kinds[k] ||= { points: 0, hits: 0 }; o.points++; if (hit) o.hits++; };
  let tracksFound = 0, tracksHalf = 0, latencySum = 0, fragments = 0;
  const perTrack = [];
  for (const tr of gt.tracks) {
    const pts = tr.pts.filter(p => p[0] >= f0 && p[0] < f1 && p[0] % SKIP === 0);
    if (!pts.length) continue;
    let hits = 0, first = -1;
    const ids = new Set();
    pts.forEach((p, i) => {
      const [t, cx, cy, w = 0, h = 0] = p;
      const span = Math.max(1, Math.round(4 / SKIP));
      const a = pts[Math.max(0, i - span)], b = pts[Math.min(pts.length - 1, i + span)];
      const moving = Math.hypot(b[1] - a[1], b[2] - a[2]) / Math.max(1, b[0] - a[0]) >= MOVING_PX_PER_FRAME;
      const r = Math.max(HIT_PX, 0.6 * Math.max(w, h));
      let hit = false;
      for (const o of shownAt(t)) if (Math.hypot(o.cx - cx, o.cy - cy) <= r) { hit = true; ids.add(o.id); }
      if (hit) { hits++; if (first < 0) first = i; }
      bump(tr.kind === 'slow' ? (moving ? 'slow' : 'slowPaused') : tr.kind, hit);
      bump('all', hit);
    });
    if (hits >= Math.min(3, pts.length)) { tracksFound++; latencySum += first; fragments += ids.size; }
    if (hits >= pts.length / 2) tracksHalf++;
    perTrack.push({ id: tr.id, kind: tr.kind, points: pts.length, hits, firstHit: first, ids: ids.size });
  }

  // False positives: outputs no listed object explains.
  const explain = new Map();   // t -> [[cx, cy, r]]
  const add = (t, cx, cy, w, h) => { let a = explain.get(t); if (!a) explain.set(t, a = []); a.push([cx, cy, Math.max(DC_PX, 0.75 * Math.max(w || 0, h || 0))]); };
  for (const tr of [...gt.tracks, ...(gt.ignoreTracks || [])]) for (const [t, cx, cy, w, h] of tr.pts) add(t, cx, cy, w, h);
  const regions = gt.ignoreRegions || [];
  const explained = (o, t) => {
    for (let dt = -2 * SKIP; dt <= 2 * SKIP; dt++) for (const [cx, cy, r] of explain.get(t + dt) || []) if (Math.hypot(o.cx - cx, o.cy - cy) <= r) return true;
    for (const g of regions) if (t >= g.t0 && t <= g.t1 && o.cx >= g.x - 10 && o.cx <= g.x + g.w + 10 && o.cy >= g.y - 10 && o.cy <= g.y + g.h + 10) return true;
    return false;
  };
  let outputs = 0, fp = 0, framesWithOutput = 0;
  const byId = new Map();
  for (const [t, list] of outs) {
    if (t < f0) continue;
    if (list.length) framesWithOutput++;
    for (const o of list) {
      outputs++;
      const ok = explained(o, t);
      if (!ok) fp++;
      const s = byId.get(o.id) || { n: 0, ok: 0 };
      s.n++; if (ok) s.ok++;
      byId.set(o.id, s);
    }
  }
  let falseTracks = 0;
  for (const s of byId.values()) if (s.ok < 0.3 * s.n) falseTracks++;
  const frames = Math.max(1, processed - Math.ceil(f0 / SKIP));
  const rate = (k) => kinds[k] ? +(kinds[k].hits / kinds[k].points).toFixed(3) : null;
  return {
    clip: name, exhaustive: !!gt.exhaustive, frames,
    gtTracks: perTrack.length, gtPoints: kinds.all?.points || 0,
    recall: rate('all'), recallFast: rate('fast'), recallSlow: rate('slow'), recallSlowPaused: rate('slowPaused'), recallBig: rate('big'),
    points: Object.fromEntries(Object.entries(kinds).map(([k, v]) => [k, v.points])),
    tracksFound, tracksHalf,
    latency: tracksFound ? +(latencySum / tracksFound).toFixed(1) : null,
    idsPerTrack: tracksFound ? +(fragments / tracksFound).toFixed(2) : null,
    outputsPerFrame: +(outputs / frames).toFixed(3),
    fpPerFrame: +(fp / frames).toFixed(3),
    falseTracksPerMin: +(falseTracks / (frames * SKIP / meta.fps / 60)).toFixed(1),
    framesWithOutput: +(framesWithOutput / frames).toFixed(3),
    msPerFrame: +(ms / Math.max(1, processed)).toFixed(2),
    perTrack,
  };
}

// ---------------------------------------------------------------- synthetic flies
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

const STRATA = {
  // Erratic fliers: 4–28 px/frame, 2.5–7 px across, blurred along their path.
  fast: { count: 5, seed: 7, speed: [4, 28], radius: [1.25, 3.5], alpha: [0.5, 0.95], jitter: 6, wobble: 1.2, turn: 0.08, pause: 0, blur: true },
  // Walkers / hoverers: 0.2–2 px/frame, 4–9 px across, sharp, with rests.
  slow: { count: 5, seed: 11, speed: [0.2, 2], radius: [2, 4.5], alpha: [0.6, 0.95], jitter: 0.25, wobble: 0.5, turn: 0.02, pause: 0.012, blur: false },
};

/** Returns paint(frame) → truth [{x, y, r, speed, visible, paused}] for that frame. */
function flyPainter(cfg) {
  const R = random(cfg.seed);
  const between = ([a, b]) => a + R() * (b - a);
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
    const spawn = () => ({ x: 40 + R() * (f.width - 80), y: 40 + R() * (f.height - 80), dir: R() * 6.283, speed: between(cfg.speed), r: between(cfg.radius), alpha: between(cfg.alpha), rest: 0 });
    if (!flies) flies = Array.from({ length: cfg.count }, spawn);
    const truth = [];
    for (let k = 0; k < flies.length; k++) {
      const fly = flies[k];
      fly.dir += (R() - 0.5) * cfg.wobble;
      if (R() < cfg.turn) fly.dir += (R() - 0.5) * 4;       // sudden turn
      fly.speed = Math.min(cfg.speed[1], Math.max(cfg.speed[0], fly.speed + (R() - 0.5) * cfg.jitter));
      if (fly.rest > 0) fly.rest--;
      else if (R() < cfg.pause) fly.rest = 10 + ((R() * 40) | 0);
      const v = fly.rest > 0 ? 0 : fly.speed;
      const px = fly.x, py = fly.y;
      fly.x += Math.cos(fly.dir) * v;
      fly.y += Math.sin(fly.dir) * v;
      if (fly.x < 10 || fly.y < 10 || fly.x > f.width - 10 || fly.y > f.height - 10) { flies[k] = spawn(); continue; }
      const visible = luminance(f, fly.x, fly.y) > 100;   // a dark fly only shows on a bright background
      const steps = cfg.blur ? Math.max(1, Math.ceil(v * 0.5 / Math.max(1, fly.r))) : 1;   // blur over half a frame of travel
      for (let s = 0; s < steps; s++) {
        const t = 1 - 0.5 * s / steps;
        stamp(f, px + (fly.x - px) * t, py + (fly.y - py) * t, fly.r, fly.alpha / Math.sqrt(steps));
      }
      truth.push({ x: fly.x, y: fly.y, r: fly.r, speed: v, visible, paused: fly.rest > 0 });
    }
    return truth;
  };
}

function scoreSynth(name, stratum) {
  const meta = clipMeta(name);
  const step = makePipeline();
  // Every frame is painted (so a skipped frame still advances the flies); only every SKIPth is analysed.
  const cfg = STRATA[stratum];
  const size = meta.w * meta.h * 4, data = new Uint8ClampedArray(size), frame = { data, width: meta.w, height: meta.h };
  const paintFn = flyPainter(cfg);
  const fd = fs.openSync(meta.raw, 'r');
  let visible = 0, found = 0, pausedVisible = 0, pausedFound = 0, marks;
  try {
    for (let t = 0; t < Math.min(SYNTH_FRAMES, meta.n); t++) {
      fs.readSync(fd, data, 0, size, t * size);
      const truth = paintFn(frame);
      if (t % SKIP) continue;                               // not analysed; the flies still moved
      marks = step(frame).map(d => ({ cx: d.x + d.w / 2, cy: d.y + d.h / 2, half: Math.max(d.w, d.h) / 2 }));
      if (t < 8) continue;                                  // let the tracker warm up
      for (const g of truth) {
        if (!g.visible) continue;
        const reach = 10 + 2 * g.r + g.speed * 0.5;
        const hit = marks.some(d => Math.hypot(d.cx - g.x, d.cy - g.y) < reach + d.half);
        if (g.paused) { pausedVisible++; if (hit) pausedFound++; } else { visible++; if (hit) found++; }
      }
    }
  } finally { fs.closeSync(fd); }
  return { recall: visible ? +(found / visible).toFixed(3) : null, recallPaused: pausedVisible ? +(pausedFound / pausedVisible).toFixed(3) : null };
}

// ---------------------------------------------------------------- run
const mean = (xs) => { const v = xs.filter(x => x !== null && x !== undefined && !Number.isNaN(x)); return v.length ? +(v.reduce((a, b) => a + b, 0) / v.length).toFixed(3) : null; };
const pct = (x) => x === null || x === undefined ? '   –' : (x * 100).toFixed(0).padStart(3) + '%';
const result = { motion: path.relative(ROOT, motionPath), tracker: path.relative(ROOT, trackerPath), mopt: motionOptions, topt: trackerOptions, skip: SKIP, maxside: MAXSIDE, real: [], synth: [] };

if (!has('no-real')) {
  console.log('REAL (tools/gt)          fast  slow  paused  big | tracks found  latency  ids/trk | out/frm  FP/frm  falseTrk/min | ms/frm');
  for (const name of clips) {
    const r = scoreReal(name);
    if (!r) { console.log(name.padEnd(24) + ' (no ground truth)'); continue; }
    result.real.push(r);
    console.log(`${(name + (r.exhaustive ? '' : ' *')).padEnd(24)} ${pct(r.recallFast)}  ${pct(r.recallSlow)}   ${pct(r.recallSlowPaused)}  ${pct(r.recallBig)} | ` +
      `${String(r.tracksFound).padStart(3)}/${String(r.gtTracks).padEnd(3)}      ${String(r.latency ?? '–').padStart(5)}    ${String(r.idsPerTrack ?? '–').padStart(5)} | ` +
      `${r.outputsPerFrame.toFixed(2).padStart(6)}  ${r.fpPerFrame.toFixed(2).padStart(6)}  ${r.falseTracksPerMin.toFixed(0).padStart(8)}     | ${r.msPerFrame.toFixed(1).padStart(5)}`);
  }
  if (result.real.some(r => !r.exhaustive)) console.log('  * ground truth not exhaustive there: FP figures are an upper bound and stay out of the summary');
}
if (!has('no-synth')) {
  console.log('SYNTH (painted flies)    fast  slow  slow-paused');
  for (const name of clips.filter(c => SYNTH_CLIPS.includes(c))) {
    const fast = scoreSynth(name, 'fast'), slow = scoreSynth(name, 'slow');
    result.synth.push({ clip: name, fast: fast.recall, slow: slow.recall, slowPaused: slow.recallPaused });
    console.log(`${name.padEnd(24)} ${pct(fast.recall)}  ${pct(slow.recall)}   ${pct(slow.recallPaused)}`);
  }
}

const ex = result.real.filter(r => r.exhaustive);
const summary = {
  realFast: mean(result.real.map(r => r.recallFast)),
  realSlow: mean(result.real.map(r => r.recallSlow)),
  realSlowPaused: mean(result.real.map(r => r.recallSlowPaused)),
  realBig: mean(result.real.map(r => r.recallBig)),
  synthFast: mean(result.synth.map(r => r.fast)),
  synthSlow: mean(result.synth.map(r => r.slow)),
  synthSlowPaused: mean(result.synth.map(r => r.slowPaused)),
  fpPerFrame: mean(ex.map(r => r.fpPerFrame)),
  falseTracksPerMin: mean(ex.map(r => r.falseTracksPerMin)),
  msPerFrame: mean(result.real.map(r => r.msPerFrame)),
};
// One number for quick comparisons. Recall terms that could not be measured
// (clip set without that kind) drop out and the weights are renormalised.
const terms = [[summary.realFast, 0.25], [summary.realSlow, 0.25], [summary.realBig, 0.1], [summary.synthFast, 0.15], [summary.synthSlow, 0.2], [summary.realSlowPaused, 0.05]].filter(([v]) => v !== null);
const weight = terms.reduce((a, [, w]) => a + w, 0);
summary.score = terms.length ? +(100 * terms.reduce((a, [v, w]) => a + v * w, 0) / weight - 25 * (summary.fpPerFrame ?? 0)).toFixed(1) : null;
result.summary = summary;
console.log('SUMMARY ' + JSON.stringify(summary));
if (opt('json')) fs.writeFileSync(opt('json'), JSON.stringify(result, null, 1));
