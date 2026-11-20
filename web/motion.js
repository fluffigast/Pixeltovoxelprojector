// motion.js — Simple two-frame differencing for flying object detection
//
// Matches the core algorithm from ray_voxel.cpp:
//   diff = |current_frame - previous_frame|
//   if diff > threshold → motion pixel
//
// Then: connected components → blob centroids → output detections

export class MotionDetector {
  constructor(options = {}) {
    this.threshold = options.threshold ?? 20;
    this.minArea = options.minArea ?? 3;
    this.maxArea = options.maxArea ?? 2000;
    this.prevGray = null;
  }

  setThreshold(v) { this.threshold = v; }

  static toGrayscale(imageData) {
    const { data, width, height } = imageData;
    const gray = new Uint8Array(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }
    return gray;
  }

  // Simple two-frame absolute difference — same as ray_voxel.cpp detect_motion()
  _frameDiff(gray, w, h) {
    const n = w * h;
    const result = new Uint8Array(n);

    if (!this.prevGray) {
      this.prevGray = gray;
      return result;
    }

    for (let i = 0; i < n; i++) {
      const d = gray[i] - this.prevGray[i];
      result[i] = d > 0 ? d : -d;
    }

    this.prevGray = gray;
    return result;
  }

  // Threshold + connected components + centroids
  _detect(diff, w, h) {
    const t = this.threshold;
    const n = w * h;

    // Threshold to binary mask
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = diff[i] > t ? 1 : 0;

    // Connected components (union-find)
    const labels = new Int32Array(n);
    const parent = [0];
    let next = 1;
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[Math.max(a, b)] = Math.min(a, b); };

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        const above = y > 0 ? labels[i - w] : 0;
        const left = x > 0 ? labels[i - 1] : 0;
        if (above && left) { labels[i] = Math.min(above, left); union(above, left); }
        else if (above) labels[i] = above;
        else if (left) labels[i] = left;
        else { labels[i] = next; parent.push(next); next++; }
      }
    }
    for (let i = 0; i < n; i++) if (labels[i]) labels[i] = find(labels[i]);

    // Extract blobs
    const blobs = new Map();
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const lbl = labels[y * w + x];
        if (!lbl) continue;
        let b = blobs.get(lbl);
        if (!b) { b = { sx: 0, sy: 0, area: 0, minX: x, minY: y, maxX: x, maxY: y }; blobs.set(lbl, b); }
        b.sx += x; b.sy += y; b.area++;
        if (x < b.minX) b.minX = x;
        if (x > b.maxX) b.maxX = x;
        if (y < b.minY) b.minY = y;
        if (y > b.maxY) b.maxY = y;
      }
    }

    const dets = [];
    for (const b of blobs.values()) {
      if (b.area < this.minArea || b.area > this.maxArea) continue;
      dets.push({
        cx: b.sx / b.area, cy: b.sy / b.area,
        w: b.maxX - b.minX + 1, h: b.maxY - b.minY + 1,
        area: b.area
      });
    }
    return dets;
  }

  update(imageData) {
    const { width: w, height: h } = imageData;
    const gray = MotionDetector.toGrayscale(imageData);
    const diff = this._frameDiff(gray, w, h);

    if (!this.prevGray && diff.every(v => v === 0)) return [];

    const dets = this._detect(diff, w, h);

    const results = [];
    for (const d of dets) {
      results.push({
        x: Math.round(d.cx - d.w / 2),
        y: Math.round(d.cy - d.h / 2),
        w: d.w, h: d.h, area: d.area,
        label: '', confidence: -1, source: 'motion'
      });
    }
    results.sort((a, b) => b.area - a.area);
    return results.slice(0, 50);
  }
}
