// tracker.js — Constant-velocity nearest-neighbour tracker with confirmation.
//
// A blob becomes a visible track only after it has been matched on `minHits`
// consecutive frames (single-frame noise never reaches the screen). Matching
// predicts each track one frame ahead from its velocity, so fast flies
// (30+ px/frame) are followed without widening the gate for everything else.
//
// That one-frame prediction is also the aim point, so its quality matters:
//  - Position is the detection's weighted centroid when it has one (`cx`, `cy`
//    from motion.js). The centre of the bounding box jumps by a pixel or two
//    whenever a wing or a stray pixel changes the box, and that noise is
//    doubled in the velocity.
//  - Velocity is kept two ways: the last step alone (follows a turn at once,
//    but carries all the position noise) and a smoothed one (steadier, but
//    late in a turn). Each track uses whichever has predicted it better over
//    its last few frames: the smoothed one for a cruising insect, the last
//    step for one that darts about.

/** Rough class guess for motion blobs, which carry no label of their own. */
export function classifyBySize(w, h) {
  const area = w * h;
  if (area < 900) return 'insect';
  if (area < 90000) return 'unknown';
  return 'drone';
}

const TRAIL_LEN = 16;   // recent centres kept per track, drawn as a flight path
const SMOOTH_BLEND = 0.5;   // share of each new step taken into the smoothed velocity
const ERROR_MEMORY = 0.7;   // how much of a predictor's past error is kept per frame
const SMOOTH_MARGIN = 1.1;  // the smoothed velocity must be this much better to be used

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
   * @returns {Array} one entry per confirmed live track: detection shape plus `id`, `missed`, `vx`, `vy`,
   *   `aimX`/`aimY` (where it is expected next frame),
   *   and how good the aim was this frame. The aim point is where the track was predicted to
   *   be; `aimError` is its distance (px) from where the object was actually found, and
   *   `onTarget` is true when the aim point lies inside the object's box. Both are null/false
   *   for a frame in which the object was not found, and for a track's first frame.
   */
  update(detections, _shift = null, _masks = []) {   // extra args accepted for app.js compatibility
    const tracks = this.tracks;
    const centres = detections.map(d => (d.cx != null && d.cy != null
      ? { cx: d.cx, cy: d.cy }
      : { cx: d.x + d.w / 2, cy: d.y + d.h / 2 }));

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
      // Aim check, before the track is moved: predicted position vs where it really is.
      const aimX = t.cx + t.vx, aimY = t.cy + t.vy;
      const boxX = det.x + det.w / 2, boxY = det.y + det.h / 2;
      t.aimError = Math.hypot(cx - aimX, cy - aimY);
      t.onTarget = Math.abs(aimX - boxX) <= Math.max(1, det.w / 2) && Math.abs(aimY - boxY) <= Math.max(1, det.h / 2);

      const gap = t.missed + 1;                       // frames since last observation
      const stepX = (cx - t.cx) / gap, stepY = (cy - t.cy) / gap;
      if (t.hits >= 2) {
        // Score both velocity estimates on this frame, then update them.
        const lastErr = Math.hypot(cx - (t.cx + t.lastVx), cy - (t.cy + t.lastVy));
        const smoothErr = Math.hypot(cx - (t.cx + t.smoothVx), cy - (t.cy + t.smoothVy));
        t.lastErr = t.lastErr === null ? lastErr : t.lastErr * ERROR_MEMORY + lastErr * (1 - ERROR_MEMORY);
        t.smoothErr = t.smoothErr === null ? smoothErr : t.smoothErr * ERROR_MEMORY + smoothErr * (1 - ERROR_MEMORY);
        t.smoothVx += (stepX - t.smoothVx) * SMOOTH_BLEND;
        t.smoothVy += (stepY - t.smoothVy) * SMOOTH_BLEND;
      } else {
        t.smoothVx = stepX;
        t.smoothVy = stepY;
      }
      t.lastVx = stepX;
      t.lastVy = stepY;
      const smooth = t.smoothErr !== null && t.smoothErr * SMOOTH_MARGIN < t.lastErr;
      t.vx = smooth ? t.smoothVx : stepX;
      t.vy = smooth ? t.smoothVy : stepY;
      t.cx = cx;
      t.cy = cy;
      t.boxDx = boxX - cx;                            // where the box sits relative to the centroid
      t.boxDy = boxY - cy;
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
      t.aimError = null;                              // not found this frame: aim unknown
      t.onTarget = false;
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
        boxDx: det.x + det.w / 2 - centres[di].cx, boxDy: det.y + det.h / 2 - centres[di].cy,
        lastVx: 0, lastVy: 0, smoothVx: 0, smoothVy: 0, lastErr: null, smoothErr: null,
        label: det.label || classifyBySize(det.w, det.h),
        classified,
        confidence: det.confidence, source: det.source,
        missed: 0, hits: 1, age: 1,
        aimError: null, onTarget: false,
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
        x: Math.round(t.cx + t.boxDx - t.w / 2),
        y: Math.round(t.cy + t.boxDy - t.h / 2),
        w: t.w, h: t.h,
        vx: t.vx, vy: t.vy,
        aimX: t.cx + t.vx, aimY: t.cy + t.vy,          // where it is expected next frame
        label: t.label, confidence: t.confidence, source: t.source,
        missed: t.missed,
        aimError: t.aimError, onTarget: t.onTarget,
        trail: t.trail,
      });
    }
    return out;
  }
}
