// Headless evaluation of web/motion.js + web/tracker.js on a raw RGBA frame
// dump, scored against an offline reference (tools/build_gt.py output).
//
// Frames: tools/dump_frames.py.  Reference: tools/build_gt.py (then prune by eye).
// Usage: node tools/motion_eval.js <frames.rgba> <meta.json> <gt.json>
//            [--motion <path to motion.js>] [--thr N] [--dump out.json]
//
// Metrics:
//   recall      fraction of reference track points with a raw motion blob
//                centre within HIT_PX
//   trk-recall  same, but against tracker output (what the UI shows)
//   fp/frame    tracker outputs per frame that are not within DC_PX of any
//                reference detection (reference dets of any length are a
//                don't-care zone, since the reference is not exhaustive)
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const HIT_PX = 20, DC_PX = 30;
const args = process.argv.slice(2);
const pos = args.filter((a, i) => !a.startsWith('--') && !(args[i - 1] || '').startsWith('--'));
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const [rgbaPath, metaPath, gtPath] = pos;
const motionPath = path.resolve(opt('motion', 'web/motion.js'));
const trackerPath = path.resolve(opt('tracker', 'web/tracker.js'));
const thr = Number(opt('thr', 20));
const extra = JSON.parse(opt('opt', '{}'));
const textra = JSON.parse(opt('topt', '{}'));  // extra Tracker options   // extra MotionDetector options, e.g. --opt '{"activityMax":1}'

const { MotionDetector } = await import(pathToFileURL(motionPath));
const { Tracker } = await import(pathToFileURL(trackerPath));
const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
const gt = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
const { w, h, n } = meta;
const frameBytes = w * h * 4;
const fd = fs.openSync(rgbaPath, 'r');

const det = new MotionDetector({ threshold: thr, ...extra });
const trk = new Tracker(textra);
const perFrameBlobs = [], perFrameTracks = [];
let totalMs = 0;
const buf = new Uint8ClampedArray(frameBytes);
for (let t = 0; t < n; t++) {
  fs.readSync(fd, buf, 0, frameBytes, t * frameBytes);
  const img = { data: buf, width: w, height: h };
  const t0 = performance.now();
  const blobs = det.update(img);
  const tracks = trk.update(blobs);
  totalMs += performance.now() - t0;
  perFrameBlobs.push(blobs.map(b => [b.x + b.w / 2, b.y + b.h / 2, b.w, b.h]));
  perFrameTracks.push(tracks.map(b => [b.x + b.w / 2, b.y + b.h / 2, b.w, b.h, b.label, b.id]));
}

const near = (list, x, y, r) => list.some(p => Math.hypot(p[0] - x, p[1] - y) <= r);
let gtPts = 0, hitBlob = 0, hitTrk = 0;
for (const tr of gt.tracks) for (const [t, cx, cy, gw = 0, gh = 0] of tr) {
  gtPts++;
  const r = Math.max(HIT_PX, 0.6 * Math.max(gw, gh));   // big objects: centroid may sit on a rotor, not the body
  if (near(perFrameBlobs[t], cx, cy, r)) hitBlob++;
  if (near(perFrameTracks[t], cx, cy, r)) hitTrk++;
}
let fp = 0, outs = 0, blobsTotal = 0;
for (let t = 0; t < n; t++) {
  blobsTotal += perFrameBlobs[t].length;
  for (const p of perFrameTracks[t]) {
    outs++;
    if (!near(gt.alldets[t] || [], p[0], p[1], DC_PX)) fp++;
  }
}
const res = {
  motion: path.relative(process.cwd(), motionPath), thr, opt: extra, topt: textra,
  gtPoints: gtPts, recall: +(hitBlob / gtPts).toFixed(3), trkRecall: +(hitTrk / gtPts).toFixed(3),
  blobsPerFrame: +(blobsTotal / n).toFixed(2), outputsPerFrame: +(outs / n).toFixed(2),
  fpPerFrame: +(fp / n).toFixed(2), msPerFrame: +(totalMs / n).toFixed(2),
};
console.log(JSON.stringify(res));
if (opt('dump')) fs.writeFileSync(opt('dump'), JSON.stringify({ blobs: perFrameBlobs, tracks: perFrameTracks }));
