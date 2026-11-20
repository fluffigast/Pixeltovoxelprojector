// tracker.js — Nearest-centroid tracker: snap to detections, brief persistence

/** Rough class guess for motion blobs, which carry no label of their own. */
export function classifyBySize(w, h) {
  const area = w * h;
  if (area < 900) return 'insect';
  if (area < 90000) return 'unknown';
  return 'drone';
}

export class Tracker {
  constructor(options = {}) {
    this.maxMatchDist = options.maxMatchDist ?? 80;   // px between centres
    this.maxMissed = options.maxMissed ?? 3;          // frames a track survives unmatched
    this.tracks = new Map();
    this._nextId = 1;
  }

  reset() {
    this.tracks.clear();
  }

  /**
   * @param {Array<{x,y,w,h,label,confidence,source}>} detections
   * @returns {Array} one entry per live track, in the same shape plus `id` and `missed`
   */
  update(detections) {
    const tracks = this.tracks;
    const centres = detections.map(d => ({ cx: d.x + d.w / 2, cy: d.y + d.h / 2 }));

    // All detection–track pairs within range, closest first, assigned greedily.
    const pairs = [];
    for (let di = 0; di < detections.length; di++) {
      const { cx, cy } = centres[di];
      for (const [id, t] of tracks) {
        const dist = Math.hypot(cx - t.cx, cy - t.cy);
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
      // Snap straight to the detection — no smoothing, no drift.
      t.cx = centres[di].cx;
      t.cy = centres[di].cy;
      t.w = det.w;
      t.h = det.h;
      t.confidence = det.confidence;
      t.source = det.source;
      t.missed = 0;
      t.age++;

      if (det.source === 'yolo' && det.label) {
        t.label = det.label;
        t.classified = true;
      } else if (!t.classified) {
        // A model label outranks the size heuristic, so only guess while the
        // track has never been classified by the model.
        t.label = classifyBySize(det.w, det.h);
      }
    }

    // Unmatched tracks age out quickly — no coasting, no ghost markers.
    for (const [id, t] of tracks) {
      if (usedTracks.has(id)) continue;
      if (++t.missed > this.maxMissed) tracks.delete(id);
    }

    // Unmatched detections start new tracks.
    for (let di = 0; di < detections.length; di++) {
      if (usedDets.has(di)) continue;
      const det = detections[di];
      const classified = det.source === 'yolo' && !!det.label;
      tracks.set(this._nextId++, {
        cx: centres[di].cx, cy: centres[di].cy, w: det.w, h: det.h,
        label: det.label || classifyBySize(det.w, det.h),
        classified,
        confidence: det.confidence, source: det.source,
        missed: 0, age: 1,
      });
    }

    const out = [];
    for (const [id, t] of tracks) {
      out.push({
        id,
        x: Math.round(t.cx - t.w / 2),
        y: Math.round(t.cy - t.h / 2),
        w: t.w, h: t.h,
        label: t.label, confidence: t.confidence, source: t.source,
        missed: t.missed,
      });
    }
    return out;
  }
}
