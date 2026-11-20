// motion.js — Two-frame differencing for small flying-object detection
//
// Same core algorithm as ray_voxel.cpp detect_motion():
//   diff = |current_frame - previous_frame|;  diff > threshold → motion pixel
// followed by 4-connected component labelling and a blob size filter.
//
// Grayscale conversion, differencing, thresholding and labelling happen in a
// single pass over the frame, and every buffer is reused between frames, so a
// steady-state update() allocates nothing but the (small) result array.

const INITIAL_LABELS = 4096;

export class MotionDetector {
  constructor(options = {}) {
    this.threshold = options.threshold ?? 20;
    this.minArea = options.minArea ?? 3;
    this.maxArea = options.maxArea ?? 2000;
    this.maxBlobs = options.maxBlobs ?? 50;
    // If more than this fraction of the frame changed at once it is a scene
    // cut, exposure jump or camera move rather than flying objects: report nothing.
    this.maxMotionFraction = options.maxMotionFraction ?? 0.25;

    this._n = 0;
    this._gray = null;      // current frame (swapped with _prev every update)
    this._prev = null;
    this._hasPrev = false;
    this._labels = null;    // per-pixel provisional label, 0 = background
    this._allocLabels(INITIAL_LABELS);
  }

  setThreshold(v) { this.threshold = v; }

  /** Forget the previous frame (call when the video source changes). */
  reset() { this._hasPrev = false; }

  _allocLabels(cap) {
    const grow = (old, Type) => {
      const arr = new Type(cap);
      if (old) arr.set(old);
      return arr;
    };
    this._parent = grow(this._parent, Int32Array);
    this._area = grow(this._area, Int32Array);
    this._minX = grow(this._minX, Int32Array);
    this._maxX = grow(this._maxX, Int32Array);
    this._minY = grow(this._minY, Int32Array);
    this._maxY = grow(this._maxY, Int32Array);
    this._cap = cap;
  }

  _ensureFrameBuffers(n) {
    if (this._n === n) return;
    this._n = n;
    this._gray = new Uint8Array(n);
    this._prev = new Uint8Array(n);
    this._labels = new Int32Array(n);
    this._hasPrev = false;
  }

  /**
   * @param {ImageData} imageData RGBA frame
   * @returns {Array<{x,y,w,h,area,label,confidence,source}>} blobs, largest first
   */
  update(imageData) {
    const { data, width: w, height: h } = imageData;
    const n = w * h;
    this._ensureFrameBuffers(n);

    const gray = this._gray;
    const prev = this._prev;

    // First frame after a reset: just store it.
    if (!this._hasPrev) {
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        gray[i] = (data[j] * 77 + data[j + 1] * 150 + data[j + 2] * 29) >> 8;
      }
      this._gray = prev;
      this._prev = gray;
      this._hasPrev = true;
      return [];
    }

    const t = this.threshold;
    const labels = this._labels;
    let parent = this._parent;
    let area = this._area;
    let minX = this._minX, maxX = this._maxX, minY = this._minY, maxY = this._maxY;
    let next = 1;          // next unused label
    let motionPixels = 0;

    for (let y = 0, i = 0, j = 0; y < h; y++) {
      for (let x = 0; x < w; x++, i++, j += 4) {
        const g = (data[j] * 77 + data[j + 1] * 150 + data[j + 2] * 29) >> 8;
        gray[i] = g;
        const d = g - prev[i];
        if ((d < 0 ? -d : d) <= t) { labels[i] = 0; continue; }

        motionPixels++;
        let a = y > 0 ? labels[i - w] : 0;   // label above
        let b = x > 0 ? labels[i - 1] : 0;   // label to the left
        let lbl;
        if (a && b) {
          // Union-find with path halving; the smaller label becomes the root.
          while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
          while (parent[b] !== b) { parent[b] = parent[parent[b]]; b = parent[b]; }
          if (a < b) { parent[b] = a; lbl = a; } else { parent[a] = b; lbl = b; }
        } else if (a || b) {
          lbl = a || b;
        } else {
          if (next === this._cap) {
            this._allocLabels(this._cap * 2);
            parent = this._parent; area = this._area;
            minX = this._minX; maxX = this._maxX; minY = this._minY; maxY = this._maxY;
          }
          lbl = next++;
          parent[lbl] = lbl;
          area[lbl] = 0;
          minX[lbl] = x; maxX[lbl] = x; minY[lbl] = y; maxY[lbl] = y;
        }
        labels[i] = lbl;
        area[lbl]++;
        if (x < minX[lbl]) minX[lbl] = x; else if (x > maxX[lbl]) maxX[lbl] = x;
        if (y > maxY[lbl]) maxY[lbl] = y;   // rows only increase, minY is set at creation
      }
    }

    // Current frame becomes the previous one; the old buffer is reused next time.
    this._gray = prev;
    this._prev = gray;

    if (motionPixels > n * this.maxMotionFraction) return [];

    // Fold every provisional label's stats into its root. Each label holds only
    // its own pixels, so the visiting order does not matter.
    for (let l = next - 1; l >= 1; l--) {
      let r = l;
      while (parent[r] !== r) r = parent[r];
      if (r === l) continue;
      area[r] += area[l];
      if (minX[l] < minX[r]) minX[r] = minX[l];
      if (maxX[l] > maxX[r]) maxX[r] = maxX[l];
      if (minY[l] < minY[r]) minY[r] = minY[l];
      if (maxY[l] > maxY[r]) maxY[r] = maxY[l];
    }

    const blobs = [];
    for (let l = 1; l < next; l++) {
      if (parent[l] !== l) continue;
      const a = area[l];
      if (a < this.minArea || a > this.maxArea) continue;
      blobs.push({
        x: minX[l], y: minY[l],
        w: maxX[l] - minX[l] + 1, h: maxY[l] - minY[l] + 1,
        area: a, label: '', confidence: -1, source: 'motion',
      });
    }
    blobs.sort((p, q) => q.area - p.area);
    if (blobs.length > this.maxBlobs) blobs.length = this.maxBlobs;
    return blobs;
  }
}
