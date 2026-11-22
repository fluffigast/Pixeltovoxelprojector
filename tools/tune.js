// Coordinate-descent tuner. For each parameter in order, sweep candidate
// values, run the audit harness on every sample clip, pick the value
// that maximizes mean hit rate (equal weight per clip). Iterate until
// no parameter changes value in a full pass.
//
// Score function uses mean hit rate across clips. Equal weight so a
// high-event clip like bees_hive doesn't dominate a low-event clip like
// drone_spotted — we care about generalization across scene types.

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const SAMPLES = [
  'new_video2', 'new_video', 'drone_flying_sky', 'drone_in_sky',
  'drone_spotted', 'insects_flying_forest', 'bees_hive',
];
const SAMPLES_DIR = path.join(__dirname, '..', 'web', 'samples');
const HARNESS    = path.join(__dirname, 'audit_harness.js');
const MAX_FRAMES = 400;

// Candidate values per parameter, in order of expected impact.
const GRID = {
  lead:          [20, 30, 40, 50, 60],
  rad:           [15, 20, 25, 30],
  pers:          [2, 3],
  minAge:        [3, 4, 5, 6],
  minFireSpread: [3, 4, 6, 8],
  cooldown:      [8, 12, 16],
  thr:           [6, 10, 14],
  k:             [1.5, 2.0, 2.5],
  spread:        [6, 8, 10],
  swin:          [6, 8, 10],
};

// Starting point: current defaults.
const PARAMS = {
  lead: 40, rad: 20, pers: 2, minAge: 4, minFireSpread: 4,
  cooldown: 12, thr: 10, k: 2.0, spread: 8, swin: 8,
};

function argName(k) { return '--' + k.replace(/[A-Z]/g, m => '-' + m.toLowerCase()); }

function runOne(clip, params) {
  const args = [HARNESS, path.join(SAMPLES_DIR, clip + '.mp4'), '--max-frames', String(MAX_FRAMES)];
  for (const [k, v] of Object.entries(params)) { args.push(argName(k), String(v)); }
  const r = spawnSync('node', args, { encoding: 'utf8' });
  if (r.status !== 0) {
    console.error('audit failed for', clip, ':', r.stderr?.slice(0, 200));
    return null;
  }
  try { return JSON.parse(r.stdout); } catch { return null; }
}

function score(params) {
  const perClip = [];
  for (const c of SAMPLES) {
    const r = runOne(c, params);
    if (!r) { perClip.push({ clip: c, hr: 0 }); continue; }
    perClip.push({ clip: c, hr: r.hitRate, events: r.events, p50: r.error_px.p50, p90: r.error_px.p90 });
  }
  const hrs = perClip.map(x => x.hr);
  const mean = hrs.reduce((a,b)=>a+b,0) / hrs.length;
  return { mean, perClip };
}

function fmtClip(c) {
  return c.clip.slice(0, 14).padEnd(14) + ' ' + (100 * c.hr).toFixed(1).padStart(5) + '%';
}

function pass(pname) {
  const cands = GRID[pname];
  const baseline = PARAMS[pname];
  let best = { v: baseline, s: null };
  console.log('\n== tuning', pname, '(baseline', baseline, '); candidates:', cands.join(','));
  for (const v of cands) {
    const trial = { ...PARAMS, [pname]: v };
    const s = score(trial);
    const marker = (best.s && s.mean > best.s.mean) || !best.s ? '*' : ' ';
    console.log(marker, pname, '=', String(v).padStart(5), '-> mean HR', (100*s.mean).toFixed(2) + '%',
      '[' + s.perClip.map(c => (100*c.hr).toFixed(0)).join(',') + ']');
    if (!best.s || s.mean > best.s.mean) best = { v, s };
  }
  const changed = best.v !== baseline;
  if (changed) console.log('--> UPDATE', pname, baseline, '=>', best.v, '(mean HR', (100*best.s.mean).toFixed(2) + '%)');
  else console.log('--> KEEP  ', pname, '=', baseline);
  PARAMS[pname] = best.v;
  return { changed, bestMean: best.s.mean };
}

function main() {
  const t0 = Date.now();
  console.log('Baseline score...');
  const base = score(PARAMS);
  console.log('baseline mean HR =', (100*base.mean).toFixed(2) + '%');
  for (const c of base.perClip) console.log('  ', fmtClip(c));

  const order = ['lead', 'rad', 'minFireSpread', 'minAge', 'cooldown', 'pers', 'thr', 'k', 'spread', 'swin'];
  let round = 0;
  while (round < 2) {
    round++;
    console.log('\n### Round', round);
    let anyChange = false;
    for (const p of order) {
      const r = pass(p);
      if (r.changed) anyChange = true;
    }
    if (!anyChange) break;
  }

  const final = score(PARAMS);
  console.log('\n=== FINAL ===');
  console.log('params:', JSON.stringify(PARAMS));
  console.log('mean HR:', (100 * final.mean).toFixed(2) + '% (baseline', (100 * base.mean).toFixed(2) + '%)');
  for (const c of final.perClip) console.log('  ', fmtClip(c));
  console.log('elapsed', ((Date.now() - t0) / 1000).toFixed(0) + 's');
  fs.writeFileSync('/tmp/tune_result.json', JSON.stringify({ params: PARAMS, baseline: base, final }, null, 2));
}

main();
