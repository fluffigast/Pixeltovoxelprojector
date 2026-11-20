// tracker.js — Constant-velocity nearest-neighbour tracker with confirmation.
//
// A blob becomes a visible track only after it has been matched on `minHits`
// consecutive frames (single-frame noise never reaches the screen). Matching
// predicts each track one frame ahead from its last velocity, so fast flies
// (30+ px/frame) are followed without widening the gate for everything else.

/** Rough class guess for motion blobs, which carry no label of their own. */
export function classifyBySize(w, h) {
  const area = w * h;
  if (area < 900) return 'insect';
  if (area < 90000) return 'unknown';
  return 'drone';
}

const TRAIL_LEN = 16;   // recent centres kept per track, drawn as a flight path

export class Tracker {
  constructor(options = {}) {
    this.maxMatchDist = options.maxMatchDist ?? 60;   // px from the predicted position
    this.maxMissed = options.maxMissed ?? 2;          // frames a track survives unmatched
    this.minHits = options.minHits ?? 2;              // matched frames before a track is reported
    this.tracks = new Map();
    this._nextId = 1;
  }

  reset() {
    this.tracks.clear();
  }

  /**
   * @param {Array<{x,y,w,h,label,confidence,source}>} detections
   * @returns {Array} one entry per confirmed live track: detection shape plus `id`, `missed`, `vx`, `vy`
   */
  update(detections, _shift = null, _masks = []) {   // extra args accepted for app.js compatibility
    const tracks = this.tracks;
    const centres = detections.map(d => ({ cx: d.x + d.w / 2, cy: d.y + d.h / 2 }));

    // Predicted position = last position + last velocity (coasting tracks keep extrapolating).
    const pairs = [];
    for (const [id, t] of tracks) {
      const px = t.cx + t.vx, py = t.cy + t.vy;
      for (let di = 0; di < detections.length; di++) {
        const dist = Math.hypot(centres[di].cx - px, centres[di].cy - py);
        if (dist < this.maxMatchDist) pairs.push({ di, id, dist });
      }
    }
    pairs.sort((a, b) => a.dist - b.dist);

    const usedTracks = new Set();
    const usedDets = new Set();
    for (const { di, id } of pairs) {
      if (usedTracks.has(id) || usedDets.has(di)) continue;
      usedTracks.add(id);
      usedDets.add(di);

      const det = detections[di];
      const t = tracks.get(id);
      const { cx, cy } = centres[di];
      const gap = t.missed + 1;                       // frames since last observation
      t.vx = (cx - t.cx) / gap;
      t.vy = (cy - t.cy) / gap;
      t.cx = cx;
      t.cy = cy;
      t.w = det.w;
      t.h = det.h;
      t.confidence = det.confidence;
      t.source = det.source;
      t.missed = 0;
      t.hits++;
      t.age++;
      t.trail.push([cx, cy]);
      if (t.trail.length > TRAIL_LEN) t.trail.shift();

      if (det.source === 'yolo' && det.label) {
        t.label = det.label;
        t.classified = true;
      } else if (!t.classified) {
        t.label = classifyBySize(det.w, det.h);
      }
    }

    // Unmatched tracks coast on their velocity briefly, then die.
    for (const [id, t] of tracks) {
      if (usedTracks.has(id)) continue;
      if (++t.missed > this.maxMissed) { tracks.delete(id); continue; }
      t.cx += t.vx;
      t.cy += t.vy;
    }

    // Unmatched detections start new (unconfirmed) tracks.
    for (let di = 0; di < detections.length; di++) {
      if (usedDets.has(di)) continue;
      const det = detections[di];
      const classified = det.source === 'yolo' && !!det.label;
      tracks.set(this._nextId++, {
        cx: centres[di].cx, cy: centres[di].cy, w: det.w, h: det.h, vx: 0, vy: 0,
        label: det.label || classifyBySize(det.w, det.h),
        classified,
        confidence: det.confidence, source: det.source,
        missed: 0, hits: 1, age: 1,
        trail: [[centres[di].cx, centres[di].cy]],
      });
    }

    const out = [];
    for (const [id, t] of tracks) {
      // YOLO detections are already confident; motion blobs must be seen twice.
      if (t.hits < this.minHits && !t.classified) continue;
      if (t.missed > 0 && t.hits < this.minHits + 1) continue;   // don't coast on a barely-confirmed track
      out.push({
        id,
        x: Math.round(t.cx - t.w / 2),
        y: Math.round(t.cy - t.h / 2),
        w: t.w, h: t.h,
        vx: t.vx, vy: t.vy,
        label: t.label, confidence: t.confidence, source: t.source,
        missed: t.missed,
        trail: t.trail,
      });
    }
    return out;
  }
}
