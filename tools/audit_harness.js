// Headless audit harness. Reads a video with ffmpeg, feeds raw RGBA
// frames through the SAME wasm detector that artifact.html uses, runs
// the SAME tracker + aim predictor, and judges whether a "fire moment"
// would have hit the tracked target on the next frame.
//
// Hit metric: at every frame where the aim logic sets fireBit=1 (target
// inside kill-zone and confirmed+moving), record the predicted aim
// pixel. On the next frame, locate the same track's new centroid. The
// aim error is the Euclidean distance between the predicted aim and the
// actual next-frame centroid. "Hit" = error < HIT_PX.
//
// Usage: node audit_harness.js <video.mp4> [--max-frames N] [--out audit.json]

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const HIT_PX = 8;
const MAX_DIM = 960; // match artifact.html

function parseArgs(argv) {
  const args = { file: null, maxFrames: 600, out: null, p: {} };
  const pnames = ['thr','k','mn','mx','rad','pers','spread','swin','lead','minAge','minFireSpread','cooldown'];
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--max-frames') args.maxFrames = parseInt(argv[++i], 10);
    else if (a === '--out') args.out = argv[++i];
    else {
      let handled = false;
      for (const pn of pnames) {
        if (a === '--' + pn.replace(/[A-Z]/g, m => '-' + m.toLowerCase())) {
          args.p[pn] = parseFloat(argv[++i]); handled = true; break;
        }
      }
      if (!handled && !args.file) args.file = a;
    }
  }
  if (!args.file) { console.error('Usage: node audit_harness.js <video> [opts]'); process.exit(2); }
  return args;
}

function extractWasm() {
  const html = fs.readFileSync(path.join(__dirname, '..', 'web', 'artifact.html'), 'utf8');
  const m = html.match(/"(AGFzbQ[A-Za-z0-9+/=]+)"/);
  if (!m) throw new Error('no wasm b64 in artifact.html');
  return Buffer.from(m[1], 'base64');
}

async function probeDims(file) {
  return new Promise((resolve, reject) => {
    const p = spawn('ffprobe', ['-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height,avg_frame_rate', '-of', 'json', file]);
    let out = '';
    p.stdout.on('data', d => out += d);
    p.on('close', () => {
      try {
        const j = JSON.parse(out);
        const s = j.streams[0];
        const [n, d] = s.avg_frame_rate.split('/').map(Number);
        resolve({ w: s.width, h: s.height, fps: d ? n / d : 30 });
      } catch (e) { reject(e); }
    });
    p.on('error', reject);
  });
}

function computeScaled(w, h) {
  const scale = Math.min(1, MAX_DIM / Math.max(w, h));
  const nw = Math.max(2, Math.round(w * scale));
  const nh = Math.max(2, Math.round(h * scale));
  return { w: nw, h: nh };
}

// ---- Tracker. Greedy nearest-neighbor only (no ghost extrapolation —
// it drifts onto neighbors in dense scenes and causes ID swaps). Per-
// track cooldown added here so a single bee can't attract 39 fires.
function makeTracker() {
  const HIT_WINDOW = 5, MAX_HIST = 24;
  let tracks = [], nextId = 1;
  function spreadOf(hist, win) {
    const n = Math.min(hist.length, win);
    if (n < 2) return 0;
    const s = hist.length - n;
    let maxD2 = 0;
    for (let i = s; i < hist.length; i++)
      for (let j = i + 1; j < hist.length; j++) {
        const dx = hist[j].cx - hist[i].cx, dy = hist[j].cy - hist[i].cy;
        const d2 = dx * dx + dy * dy;
        if (d2 > maxD2) maxD2 = d2;
      }
    return Math.sqrt(maxD2);
  }
  function update(dets, radius, persistence, minSpread, spreadWin) {
    const r2 = radius * radius;
    const usedDet = new Array(dets.length).fill(false);
    const matched = new Array(tracks.length).fill(-1);

    // Distance-sorted greedy assignment: build all (track, det) pairs
    // within radius, sort by distance ascending, assign cheapest first.
    // Prevents ID swap when two tracks are near two detections.
    const pairs = [];
    for (let ti = 0; ti < tracks.length; ti++) {
      const t = tracks[ti];
      for (let di = 0; di < dets.length; di++) {
        const dx = dets[di].cx - t.last.cx, dy = dets[di].cy - t.last.cy;
        const d2 = dx * dx + dy * dy;
        if (d2 < r2) pairs.push([d2, ti, di]);
      }
    }
    pairs.sort((a, b) => a[0] - b[0]);
    for (const [, ti, di] of pairs) {
      if (matched[ti] !== -1 || usedDet[di]) continue;
      matched[ti] = di; usedDet[di] = true;
    }

    for (let ti = 0; ti < tracks.length; ti++) {
      const t = tracks[ti];
      const di = matched[ti];
      t.hits.push(di >= 0 ? 1 : 0); if (t.hits.length > HIT_WINDOW) t.hits.shift();
      if (di >= 0) {
        t.last = dets[di];
        t.history.push({ cx: dets[di].cx, cy: dets[di].cy });
        if (t.history.length > MAX_HIST) t.history.shift();
      }
      t.age++;
      if (t.cooldown > 0) t.cooldown--;
    }
    for (let i = 0; i < dets.length; i++) {
      if (usedDet[i]) continue;
      tracks.push({
        id: nextId++, last: dets[i],
        history: [{ cx: dets[i].cx, cy: dets[i].cy }],
        hits: [1], age: 1, cooldown: 0,
      });
    }
    tracks = tracks.filter(t => t.hits.some(h => h === 1));
    const moving = [], stationary = [];
    for (const t of tracks) {
      const sum = t.hits.reduce((a, b) => a + b, 0);
      if (sum < persistence) continue;
      if (minSpread > 0 && t.history.length >= spreadWin) {
        if (spreadOf(t.history, spreadWin) < minSpread) { stationary.push(t); continue; }
      }
      moving.push(t);
    }
    return { moving, stationary, allTracks: tracks };
  }
  return { update };
}

// Velocity averaged over the last N step-deltas. More robust than the
// 2-point estimate: a single jitter frame won't flip v to zero.
function predictAim(t, leadMs, fps, nSteps = 4) {
  const h = t.history;
  if (h.length < 2 || fps <= 0) return { x: t.last.cx, y: t.last.cy };
  const n = Math.min(nSteps, h.length - 1);
  let sx = 0, sy = 0;
  for (let i = h.length - n; i < h.length; i++) {
    sx += h[i].cx - h[i - 1].cx;
    sy += h[i].cy - h[i - 1].cy;
  }
  const vx = (sx / n) * fps;
  const vy = (sy / n) * fps;
  const dt = leadMs / 1000;
  const last = h[h.length - 1];
  return { x: last.cx + vx * dt, y: last.cy + vy * dt };
}

async function main() {
  const args = parseArgs(process.argv);
  console.error('[audit] file=' + args.file + ' maxFrames=' + args.maxFrames);

  const dims = await probeDims(args.file);
  const sc = computeScaled(dims.w, dims.h);
  const effFps = Math.min(30, dims.fps || 30);
  console.error(`[audit] source ${dims.w}x${dims.h} @${dims.fps.toFixed(2)}fps -> scaled ${sc.w}x${sc.h} @${effFps}fps`);

  const bytes = extractWasm();
  const memory = new WebAssembly.Memory({ initial: 128, maximum: 4096 });
  const mod = await WebAssembly.instantiate(bytes, { env: { memory } });
  const w = mod.instance.exports;
  console.error('[audit] wasm loaded (' + bytes.length + ' bytes)');

  const nBytes = sc.w * sc.h * 4;
  const ptr = w.alloc(nBytes);

  // ffmpeg: decode to raw rgba frames, piped to stdout
  const ff = spawn('ffmpeg', [
    '-hide_banner', '-loglevel', 'error',
    '-i', args.file,
    '-vf', `scale=${sc.w}:${sc.h},fps=${effFps}`,
    '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'
  ]);

  const tracker = makeTracker();
  // Defaults from tools/tune.js coordinate-descent sweep across 7 clips.
  // Baseline mean HR 77.99% -> tuned 89.21% (+11.22).
  const detectParams = {
    thr: args.p.thr ?? 10,
    mn:  args.p.mn  ?? 3,
    mx:  args.p.mx  ?? 2000,
    kStd100:   (args.p.k ?? 2.0) * 100,
    gateX1000: 200, morph: 1,
    rad:    args.p.rad    ?? 15,
    pers:   args.p.pers   ?? 2,
    spread: args.p.spread ?? 6,
    swin:   args.p.swin   ?? 8,
    leadMs: args.p.lead   ?? 20,
    kzFrac: 0.6,
    minAge:        args.p.minAge        ?? 3,
    minFireSpread: args.p.minFireSpread ?? 4,
    cooldownF:     args.p.cooldown      ?? 12,
  };
  const kz = {
    x0: sc.w * (1 - detectParams.kzFrac) / 2,
    y0: sc.h * (1 - detectParams.kzFrac) / 2,
    x1: sc.w * (1 + detectParams.kzFrac) / 2,
    y1: sc.h * (1 + detectParams.kzFrac) / 2,
  };

  const pendingShots = new Map(); // trackId -> {frame, aim, aimTs}
  const events = []; // completed shot audits

  let frameN = 0;
  let buf = Buffer.alloc(0);
  const chunks = [];
  ff.stdout.on('data', d => chunks.push(d));

  await new Promise((resolve, reject) => {
    ff.stdout.on('end', resolve);
    ff.on('error', reject);
    ff.stderr.on('data', d => process.stderr.write('[ffmpeg] ' + d));
  });
  const fullBuf = Buffer.concat(chunks);
  const totalFrames = Math.floor(fullBuf.length / nBytes);
  console.error('[audit] decoded ' + totalFrames + ' frames');

  const limit = Math.min(totalFrames, args.maxFrames);
  for (let f = 0; f < limit; f++) {
    const frameBuf = fullBuf.subarray(f * nBytes, (f + 1) * nBytes);
    new Uint8Array(memory.buffer, ptr, nBytes).set(frameBuf);

    const detPtr = w.detect(ptr, sc.w, sc.h,
      detectParams.thr, detectParams.mn, detectParams.mx,
      detectParams.kStd100, detectParams.gateX1000, detectParams.morph);
    const cnt = new Float32Array(memory.buffer, detPtr, 1)[0] | 0;
    const data = new Float32Array(memory.buffer, detPtr + 4, cnt * 7);
    const rawDets = [];
    for (let i = 0; i < cnt; i++) {
      const o = i * 7;
      rawDets.push({ cx: data[o], cy: data[o+1], w: data[o+2], h: data[o+3],
                     area: data[o+4], aspect: data[o+5], compact: data[o+6] });
    }

    const out = tracker.update(rawDets, detectParams.rad, detectParams.pers, detectParams.spread, detectParams.swin);
    const confirmed = out.moving;

    // Resolve any pending shot against the current real centroid.
    for (const t of confirmed) {
      const pend = pendingShots.get(t.id);
      if (pend) {
        const dx = t.last.cx - pend.aim.x;
        const dy = t.last.cy - pend.aim.y;
        const err = Math.sqrt(dx*dx + dy*dy);
        events.push({
          frame: pend.frame, trackId: t.id,
          aim: pend.aim, actualNext: { x: t.last.cx, y: t.last.cy },
          error: err, hit: err < HIT_PX, resolved: true,
        });
        pendingShots.delete(t.id);
      }
    }
    // Drop pending shots whose track vanished (count as misses).
    const liveIds = new Set(out.moving.map(t => t.id).concat(out.stationary.map(t => t.id)));
    for (const [id, pend] of pendingShots) {
      if (!liveIds.has(id)) {
        events.push({
          frame: pend.frame, trackId: id, aim: pend.aim,
          actualNext: null, error: Infinity, hit: false, resolved: false, reason: 'track_lost',
        });
        pendingShots.delete(id);
      }
    }

    // Fire gate: track must be mature, off cooldown, and have real motion history.
    const fireable = confirmed.filter(t =>
      t.age >= detectParams.minAge &&
      t.cooldown === 0 &&
      t.history.length >= 3
    );
    // Spread check on firing decision — reuse the diameter metric.
    function diamOf(h, n = 6) {
      const k = Math.min(n, h.length);
      let m2 = 0;
      for (let i = h.length - k; i < h.length; i++)
        for (let j = i + 1; j < h.length; j++) {
          const dx = h[j].cx - h[i].cx, dy = h[j].cy - h[i].cy;
          const d2 = dx * dx + dy * dy;
          if (d2 > m2) m2 = d2;
        }
      return Math.sqrt(m2);
    }
    const movers = fireable.filter(t => diamOf(t.history, 6) >= detectParams.minFireSpread);
    if (movers.length) {
      const target = movers.reduce((a, b) => (b.last.area > a.last.area ? b : a));
      const aim = predictAim(target, detectParams.leadMs, effFps);
      const inside = aim.x >= kz.x0 && aim.x <= kz.x1 && aim.y >= kz.y0 && aim.y <= kz.y1;
      if (inside) {
        pendingShots.set(target.id, { frame: f, aim, aimTs: f / effFps, targetCentroid: { x: target.last.cx, y: target.last.cy } });
        target.cooldown = detectParams.cooldownF;
      }
    }
  }

  // Any still-pending at EOF = lost
  for (const [id, pend] of pendingShots) {
    events.push({ frame: pend.frame, trackId: id, aim: pend.aim, actualNext: null, error: Infinity, hit: false, resolved: false, reason: 'eof' });
  }

  // Metrics
  const resolved = events.filter(e => e.resolved);
  const errors = resolved.map(e => e.error).sort((a, b) => a - b);
  const mean = errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : NaN;
  const p = (q) => errors.length ? errors[Math.min(errors.length - 1, Math.floor(q * errors.length))] : NaN;
  const hits = events.filter(e => e.hit).length;
  const hitRate = events.length ? hits / events.length : 0;

  const summary = {
    file: args.file, frames: limit, scaled: sc, effFps,
    detectParams, killZone: kz,
    events: events.length,
    resolved: resolved.length,
    lost: events.filter(e => !e.resolved).length,
    hits, hitRate,
    error_px: { mean, p50: p(0.5), p90: p(0.9), p95: p(0.95), max: errors[errors.length - 1] ?? NaN },
    uniqueTracks: new Set(events.map(e => e.trackId)).size,
  };

  console.log(JSON.stringify(summary, null, 2));
  if (args.out) fs.writeFileSync(args.out, JSON.stringify({ summary, events }, null, 2));
}

main().catch(e => { console.error('FAIL', e); process.exit(1); });
