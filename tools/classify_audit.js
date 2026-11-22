// Classify would-fire moments with a Venice vision model.
//
// Runs the detector/tracker pipeline against a video. For each unique
// track that triggers a fire event, crops a 48x48 window around the aim
// pixel from the FIRST fire frame and sends it to Venice. Reports the
// classification distribution + a corrected "target-legitimacy hit rate"
// that only counts fires on bee/fly/drone/wasp/mosquito/hornet classes.

const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const zlib = require('zlib');

const API_KEY = process.env.VENICE_API_KEY;
if (!API_KEY) { console.error('VENICE_API_KEY not set'); process.exit(2); }

const HARNESS_DIR = __dirname;
const SAMPLES_DIR = path.join(__dirname, '..', 'web', 'samples');
const MAX_DIM = 960;
const CROP = 48;

const VISION_MODEL = process.argv.includes('--model')
  ? process.argv[process.argv.indexOf('--model') + 1]
  : 'qwen-3-8-max';
const CLIP = process.argv.find(a => !a.startsWith('--') && a.endsWith('.mp4'))
  || path.join(SAMPLES_DIR, 'bees_hive.mp4');
const MAX_FRAMES = +(process.argv[process.argv.indexOf('--max-frames') + 1]) || 400;
const MAX_CLASSIFY = +(process.argv[process.argv.indexOf('--max-classify') + 1]) || 25;

const LEGITIMATE = new Set(['bee','fly','wasp','mosquito','hornet','drone','insect','flying_insect']);

function extractWasm() {
  const html = fs.readFileSync(path.join(HARNESS_DIR, '..', 'web', 'artifact.html'), 'utf8');
  return Buffer.from(html.match(/"(AGFzbQ[A-Za-z0-9+/=]+)"/)[1], 'base64');
}

async function probeDims(file) {
  return new Promise((res, rej) => {
    const p = spawn('ffprobe', ['-v','error','-select_streams','v:0','-show_entries',
      'stream=width,height,avg_frame_rate','-of','json', file]);
    let out=''; p.stdout.on('data',d=>out+=d);
    p.on('close',()=>{ try{ const j=JSON.parse(out); const s=j.streams[0]; const [n,d]=s.avg_frame_rate.split('/').map(Number); res({ w:s.width,h:s.height,fps:d?n/d:30 }); } catch(e){rej(e);} });
    p.on('error', rej);
  });
}

function computeScaled(w, h) {
  const scale = Math.min(1, MAX_DIM / Math.max(w, h));
  return { w: Math.max(2, Math.round(w * scale)), h: Math.max(2, Math.round(h * scale)) };
}

// --- Tracker (identical to audit_harness.js) ---
function makeTracker() {
  const HIT_WINDOW = 5, MAX_HIST = 24;
  let tracks = [], nextId = 1;
  function spreadOf(hist, win) {
    const n = Math.min(hist.length, win);
    if (n < 2) return 0;
    const s = hist.length - n;
    let m2 = 0;
    for (let i = s; i < hist.length; i++) for (let j = i+1; j < hist.length; j++) {
      const dx = hist[j].cx - hist[i].cx, dy = hist[j].cy - hist[i].cy;
      const d2 = dx*dx + dy*dy; if (d2 > m2) m2 = d2;
    }
    return Math.sqrt(m2);
  }
  function update(dets, radius, persistence, minSpread, spreadWin) {
    const r2 = radius * radius;
    const usedDet = new Array(dets.length).fill(false);
    const matched = new Array(tracks.length).fill(-1);
    const pairs = [];
    for (let ti = 0; ti < tracks.length; ti++) for (let di = 0; di < dets.length; di++) {
      const dx = dets[di].cx - tracks[ti].last.cx, dy = dets[di].cy - tracks[ti].last.cy;
      const d2 = dx*dx + dy*dy; if (d2 < r2) pairs.push([d2, ti, di]);
    }
    pairs.sort((a,b)=>a[0]-b[0]);
    for (const [,ti,di] of pairs) { if (matched[ti]!==-1 || usedDet[di]) continue; matched[ti]=di; usedDet[di]=true; }
    for (let ti=0; ti<tracks.length; ti++) {
      const t = tracks[ti], di = matched[ti];
      t.hits.push(di>=0?1:0); if (t.hits.length > HIT_WINDOW) t.hits.shift();
      if (di >= 0) { t.last = dets[di]; t.history.push({cx:dets[di].cx,cy:dets[di].cy}); if (t.history.length > MAX_HIST) t.history.shift(); }
      t.age++; if (t.cooldown > 0) t.cooldown--;
    }
    for (let i=0; i<dets.length; i++) if (!usedDet[i]) tracks.push({ id: nextId++, last: dets[i], history: [{cx:dets[i].cx,cy:dets[i].cy}], hits: [1], age: 1, cooldown: 0 });
    tracks = tracks.filter(t => t.hits.some(h => h === 1));
    const moving = [], stationary = [];
    for (const t of tracks) {
      const sum = t.hits.reduce((a,b)=>a+b,0); if (sum < persistence) continue;
      if (minSpread > 0 && t.history.length >= spreadWin) { if (spreadOf(t.history, spreadWin) < minSpread) { stationary.push(t); continue; } }
      moving.push(t);
    }
    return { moving, stationary };
  }
  return { update };
}

function predictAim(t, leadMs, fps, n=4) {
  const h = t.history; if (h.length < 2 || fps <= 0) return { x: t.last.cx, y: t.last.cy };
  const k = Math.min(n, h.length - 1); let sx=0, sy=0;
  for (let i = h.length - k; i < h.length; i++) { sx += h[i].cx - h[i-1].cx; sy += h[i].cy - h[i-1].cy; }
  const vx = (sx/k) * fps, vy = (sy/k) * fps, dt = leadMs/1000;
  const last = h[h.length-1]; return { x: last.cx + vx*dt, y: last.cy + vy*dt };
}

function diamOf(h, n=6) {
  const k = Math.min(n, h.length); let m2=0;
  for (let i = h.length - k; i < h.length; i++) for (let j = i+1; j < h.length; j++) {
    const dx = h[j].cx-h[i].cx, dy = h[j].cy-h[i].cy; const d2 = dx*dx+dy*dy; if (d2 > m2) m2 = d2;
  }
  return Math.sqrt(m2);
}

// --- RGBA crop -> PNG (minimal encoder) ---
function rgbaToPng(rgba, w, h) {
  const sig = Buffer.from([137,80,78,71,13,10,26,10]);
  function chunk(type, data) {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
    const typeBuf = Buffer.from(type);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
    return Buffer.concat([len, typeBuf, data, crc]);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10]=0; ihdr[11]=0; ihdr[12]=0;
  // Filter 0 per scanline
  const raw = Buffer.alloc(h * (1 + w*4));
  for (let y = 0; y < h; y++) {
    raw[y*(1+w*4)] = 0;
    rgba.copy(raw, y*(1+w*4)+1, y*w*4, (y+1)*w*4);
  }
  const idat = zlib.deflateSync(raw);
  return Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
}
let CRC_TABLE = null;
function crc32(buf) {
  if (!CRC_TABLE) {
    CRC_TABLE = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c>>>1)) : (c>>>1); CRC_TABLE[n] = c>>>0; }
  }
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = (CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c>>>8)) >>> 0;
  return (c ^ 0xffffffff) >>> 0;
}

function cropAroundAim(frameBuf, w, h, cx, cy, size) {
  const half = size >> 1;
  const x0 = Math.max(0, Math.min(w - size, Math.round(cx) - half));
  const y0 = Math.max(0, Math.min(h - size, Math.round(cy) - half));
  const out = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++) {
    const srcOff = ((y0 + y) * w + x0) * 4;
    frameBuf.copy(out, y * size * 4, srcOff, srcOff + size * 4);
  }
  return { png: rgbaToPng(out, size, size), origin: { x: x0, y: y0 } };
}

async function classify(png) {
  const b64 = png.toString('base64');
  const body = JSON.stringify({
    model: VISION_MODEL,
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text:
`Look at this small image. It is a crop around a moving object detected in a surveillance video.
Classify what you see. Reply with ONLY a JSON object, no prose:
{"class": "<one of: bee, fly, wasp, mosquito, hornet, drone, bird, leaf, debris, shadow, noise, other>",
 "confidence": <0..1>,
 "reasoning": "<short>"}` },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,' + b64 } }
      ]
    }],
    temperature: 0.0,
    max_tokens: 150,
  });
  const r = await fetch('https://api.venice.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
    body,
  });
  if (!r.ok) return { class: 'error', confidence: 0, raw: 'HTTP '+r.status };
  const j = await r.json();
  const txt = j.choices?.[0]?.message?.content ?? '';
  // Try to parse JSON block from the response
  const m = txt.match(/\{[\s\S]*\}/);
  if (!m) return { class: 'other', confidence: 0, raw: txt.slice(0, 200) };
  try { return { ...JSON.parse(m[0]), raw: txt.slice(0, 200) }; }
  catch { return { class: 'other', confidence: 0, raw: txt.slice(0, 200) }; }
}

async function main() {
  console.log('clip =', CLIP);
  console.log('model =', VISION_MODEL);
  const dims = await probeDims(CLIP);
  const sc = computeScaled(dims.w, dims.h);
  const effFps = Math.min(30, dims.fps || 30);
  const bytes = extractWasm();
  const memory = new WebAssembly.Memory({ initial: 128, maximum: 4096 });
  const mod = await WebAssembly.instantiate(bytes, { env: { memory } });
  const w = mod.instance.exports;
  const nBytes = sc.w * sc.h * 4;
  const ptr = w.alloc(nBytes);

  const P = { thr:10, mn:3, mx:2000, kStd100:200, gateX1000:200, morph:1,
    rad:15, pers:2, spread:6, swin:8, leadMs:20, kzFrac:0.6,
    minAge:3, minFireSpread:4, cooldownF:12 };
  const kz = { x0: sc.w*(1-P.kzFrac)/2, y0: sc.h*(1-P.kzFrac)/2,
               x1: sc.w*(1+P.kzFrac)/2, y1: sc.h*(1+P.kzFrac)/2 };

  const ff = spawn('ffmpeg', ['-hide_banner','-loglevel','error','-i', CLIP,
    '-vf', `scale=${sc.w}:${sc.h},fps=${effFps}`, '-f','rawvideo','-pix_fmt','rgba','-']);
  const chunks = [];
  ff.stdout.on('data', d => chunks.push(d));
  await new Promise((res,rej)=>{ ff.stdout.on('end',res); ff.on('error',rej); ff.stderr.on('data',d=>process.stderr.write('[ff] '+d)); });
  const full = Buffer.concat(chunks);
  const total = Math.floor(full.length / nBytes);
  console.log('decoded', total, 'frames at', sc.w+'x'+sc.h);

  const tracker = makeTracker();
  const firedTracks = new Map(); // trackId -> first-fire snapshot {frame, aim, png}
  const limit = Math.min(total, MAX_FRAMES);

  for (let f = 0; f < limit; f++) {
    const frame = full.subarray(f * nBytes, (f + 1) * nBytes);
    new Uint8Array(memory.buffer, ptr, nBytes).set(frame);
    const detPtr = w.detect(ptr, sc.w, sc.h, P.thr, P.mn, P.mx, P.kStd100, P.gateX1000, P.morph);
    const cnt = new Float32Array(memory.buffer, detPtr, 1)[0] | 0;
    const data = new Float32Array(memory.buffer, detPtr+4, cnt*7);
    const dets = []; for (let i=0; i<cnt; i++) { const o=i*7; dets.push({cx:data[o],cy:data[o+1],w:data[o+2],h:data[o+3],area:data[o+4],aspect:data[o+5],compact:data[o+6]}); }
    const out = tracker.update(dets, P.rad, P.pers, P.spread, P.swin);
    const fireable = out.moving.filter(t => t.age >= P.minAge && t.cooldown === 0 && t.history.length >= 3 && diamOf(t.history, 6) >= P.minFireSpread);
    if (!fireable.length) continue;
    const target = fireable.reduce((a,b)=>b.last.area > a.last.area ? b : a);
    const aim = predictAim(target, P.leadMs, effFps);
    const inside = aim.x >= kz.x0 && aim.x <= kz.x1 && aim.y >= kz.y0 && aim.y <= kz.y1;
    if (!inside) continue;
    target.cooldown = P.cooldownF;
    if (!firedTracks.has(target.id)) {
      const { png, origin } = cropAroundAim(frame, sc.w, sc.h, aim.x, aim.y, CROP);
      firedTracks.set(target.id, { frame: f, aim, origin, png, area: target.last.area });
    }
  }

  const ids = [...firedTracks.keys()];
  console.log('unique fired tracks:', ids.length, '-> classifying up to', Math.min(ids.length, MAX_CLASSIFY));
  const results = [];
  for (let i = 0; i < Math.min(ids.length, MAX_CLASSIFY); i++) {
    const id = ids[i], snap = firedTracks.get(id);
    const t0 = Date.now();
    const v = await classify(snap.png);
    const dt = Date.now() - t0;
    results.push({ trackId: id, frame: snap.frame, area: snap.area, class: v.class, confidence: v.confidence, latencyMs: dt });
    console.log(` #${id} f=${snap.frame} area=${snap.area.toFixed(0)} -> ${v.class} (conf=${v.confidence}) [${dt}ms]`);
  }

  const byClass = {};
  for (const r of results) byClass[r.class] = (byClass[r.class] || 0) + 1;
  const legitimate = results.filter(r => LEGITIMATE.has((r.class||'').toLowerCase())).length;
  const rejected   = results.filter(r => !LEGITIMATE.has((r.class||'').toLowerCase())).length;

  console.log('\n--- classification distribution ---');
  for (const [k,v] of Object.entries(byClass).sort((a,b)=>b[1]-a[1])) console.log(' ', k.padEnd(12), v);
  console.log('legitimate target fraction:', (100*legitimate/results.length).toFixed(1) + '% (' + legitimate + '/' + results.length + ')');
  console.log('reject fraction:          ', (100*rejected/results.length).toFixed(1) + '% (' + rejected + '/' + results.length + ')');
  const outFile = '/tmp/classify_' + path.basename(CLIP, '.mp4') + '.json';
  fs.writeFileSync(outFile, JSON.stringify({ clip: CLIP, model: VISION_MODEL, counts: byClass, results }, null, 2));
  console.log('wrote', outFile);
}

main().catch(e => { console.error('FAIL', e); process.exit(1); });
