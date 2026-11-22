// motion.js — Small flying-object detector for a (possibly hand-held) camera.
//
// Pipeline per frame, all on the grayscale image, no allocations in steady state:
//   1. Global shift estimate (camera shake) between the previous and the
//      current frame: coarse-to-fine integer translation search on a
//      1/16 → 1/4 → full-res pyramid, sum-of-absolute-differences.
//   2. Three-frame aligned residual:
//        d = min(|cur - prev1 shifted by s1|, |cur - prev2 shifted by s1+s2|)
//      The min kills the "ghost" a moving object leaves at its old position
//      (ordinary two-frame diff reports every object twice).
//   3. Static-edge veto: a residual pixel is dropped if both the current frame
//      and the aligned previous frame have a strong gradient there. Imperfect
//      alignment (sub-pixel shift, rotation, rolling shutter) leaves residue
//      exactly on scenery edges; a fly or drone against sky has no edge in
//      the previous frame at its new position.
//   4. Threshold → 4-connected components → area / size / fill-ratio filter.
//   5. Shake gate: if the shift is huge or too much of the frame still moves
//      after alignment, the frame is unusable; report nothing.

const INITIAL_LABELS = 4096;
const CELL_SHIFT = 5;
const CELL = 1 << CELL_SHIFT;   // 32 px; activity / cluster map resolution

export class MotionDetector {
  constructor(options = {}) {
    this.threshold = options.threshold ?? 20;
    // Per-cell adaptive threshold: max(threshold, clutterK × the cell's recent
    // mean residual). Patches that flicker every frame (foliage, water,
    // screens) raise their own floor; quiet sky stays at `threshold`.
    this.clutterK = options.clutterK ?? 6;
    this.clutterSmoothing = options.clutterSmoothing ?? 0.25;
    this.minArea = options.minArea ?? 3;
    this.maxArea = options.maxArea ?? 40000;      // a close drone can be 200 px across
    this.maxSide = options.maxSide ?? 320;        // px; larger blobs are scenery
    this.minFill = options.minFill ?? 0.25;       // area / (w*h) for small blobs; edge slivers are thin
    // A large slow object only shows its moving silhouette, so its residual
    // is sparse. Blobs wider than bigSide skip the fill test but must have
    // strong contrast (a drone body is 100+ grey levels off the sky; a merged
    // box of foliage fragments is 20-40).
    this.bigSide = options.bigSide ?? 48;
    this.bigContrast = options.bigContrast ?? 60;
    this.maxBlobs = options.maxBlobs ?? 50;
    this.edgeGrad = options.edgeGrad ?? 40;       // |gx|+|gy| counted as a strong edge
    // A residual on a static edge is kept only if it exceeds edgeK × the edge
    // strength: a few px of misalignment produces |diff| ≈ gradient, a real
    // object crossing textured background produces much more.
    this.edgeK = options.edgeK ?? 1.5;
    this.maxShift = options.maxShift ?? 40;       // px; beyond this the frame is a scene cut / violent shake
    this.maxMotionFraction = options.maxMotionFraction ?? 0.08;
    this.stabilize = options.stabilize ?? true;
    this.border = options.border ?? 8;               // px ignored at the frame edge (lens/rolling-shutter residue)
    this.shiftStep4 = options.shiftStep4 ?? 2;       // SAD sampling stride at 1/4 res
    this.shiftStepFull = options.shiftStepFull ?? 8; // SAD sampling stride at full res
    // Clutter rejection.
    //  - Background texture: a blob is kept only if the aligned previous frame
    //    is smooth around it (mean |gx|+|gy| in a margin around the box). A
    //    fly or drone against sky passes; a leaf among leaves does not.
    //  - Activity map: 1/CELL-res EMA of how often each cell produced a blob.
    //    Foliage, water, flags fire frame after frame; a fly crosses once.
    //  - Cluster veto: a cell (with its 8 neighbours) holding more than
    //    clusterMax blobs this frame is texture, not targets.
    this.bgTexMax = options.bgTexMax ?? 8;
    this.bgTexRatio = options.bgTexRatio ?? 0.3;
    this.activityDecay = options.activityDecay ?? 0.9;
    this.activityMax = options.activityMax ?? 1.0;   // 1.0 = off; 0.35 trades some recall near foliage for fewer false blobs
    this.clusterMax = options.clusterMax ?? 3;
    this.maxBlobsFrame = options.maxBlobsFrame ?? 60;   // more merged candidates than this = unusable frame
    // Fragments closer than mergeGap px are one object (a drone's body, arms
    // and rotors; a fly and its wing blur). Foliage merges into a sparse box
    // that fails the fill-ratio test instead of looking like many targets.
    this.mergeGap = options.mergeGap ?? 12;

    this.lastShift = { dx: 0, dy: 0 };
    this.lastMotionFraction = 0;

    this._n = 0; this._w = 0; this._h = 0;
    this._frames = [null, null, null];   // ring: gray buffers
    this._count = 0;                     // frames seen since reset
    this._s1 = { dx: 0, dy: 0 };         // shift prev1 → cur (most recent)
    this._s2 = { dx: 0, dy: 0 };         // shift prev2 → prev1
    this._allocLabels(INITIAL_LABELS);
  }

  setThreshold(v) { this.threshold = v; }

  /** Camera shift (prev → cur) estimated on the last frame. */
  get shift() { return this.lastShift; }

  /** Forget history (call when the video source changes). */
  reset() { this._count = 0; this._s1.dx = this._s1.dy = this._s2.dx = this._s2.dy = 0; if (this._activity) { this._activity.fill(0); this._clutter.fill(0); } }

  _allocLabels(cap) {
    const grow = (old, Type) => { const a = new Type(cap); if (old) a.set(old); return a; };
    this._parent = grow(this._parent, Int32Array);
    this._area = grow(this._area, Int32Array);
    this._sum = grow(this._sum, Int32Array);      // residual sum per label (contrast)
    this._minX = grow(this._minX, Int32Array);
    this._maxX = grow(this._maxX, Int32Array);
    this._minY = grow(this._minY, Int32Array);
    this._maxY = grow(this._maxY, Int32Array);
    this._cap = cap;
  }

  _ensureBuffers(w, h) {
    if (this._w === w && this._h === h) return;
    this._w = w; this._h = h; this._n = w * h;
    for (let i = 0; i < 3; i++) this._frames[i] = new Uint8Array(this._n);
    this._labels = new Int32Array(this._n);
    this._cw = Math.ceil(w / CELL); this._ch = Math.ceil(h / CELL);
    this._activity = new Float32Array(this._cw * this._ch);
    this._cellHits = new Uint16Array(this._cw * this._ch);
    this._cellNow = new Uint8Array(this._cw * this._ch);
    this._texHist = new Uint32Array(256);
    this._clutter = new Float32Array(this._cw * this._ch);   // EMA of mean residual per cell
    this._cellSum = new Uint32Array(this._cw * this._ch);
    this._cellN = new Uint32Array(this._cw * this._ch);
    this._cellThr = new Int32Array(this._cw * this._ch);
    // Pyramid levels for shift estimation (box-averaged).
    this._w4 = w >> 2; this._h4 = h >> 2;
    this._w16 = w >> 4; this._h16 = h >> 4;
    this._p4 = [new Uint8Array(this._w4 * this._h4), new Uint8Array(this._w4 * this._h4)];
    this._p16 = [new Uint8Array(this._w16 * this._h16), new Uint8Array(this._w16 * this._h16)];
    this._count = 0;
  }

  // Box-average downsample by 4 into dst (dst is (w>>2) x (h>>2)).
  static _down4(src, w, dst, dw, dh) {
    for (let y = 0; y < dh; y++) {
      const r0 = (y * 4) * w;
      for (let x = 0; x < dw; x++) {
        const c0 = r0 + x * 4;
        let s = 0;
        for (let yy = 0; yy < 4; yy++) {
          const r = c0 + yy * w;
          s += src[r] + src[r + 1] + src[r + 2] + src[r + 3];
        }
        dst[y * dw + x] = s >> 4;
      }
    }
  }

  // SAD between a (shifted by dx,dy) and b over the overlap, sampled every `step` px.
  // Returns mean abs diff so results are comparable across shifts.
  static _sad(a, b, w, h, dx, dy, step, margin) {
    const x0 = Math.max(margin, margin - dx), x1 = Math.min(w - margin, w - margin - dx);
    const y0 = Math.max(margin, margin - dy), y1 = Math.min(h - margin, h - margin - dy);
    let s = 0, c = 0;
    for (let y = y0; y < y1; y += step) {
      const rb = y * w, ra = (y + dy) * w + dx;
      for (let x = x0; x < x1; x += step) {
        const d = b[rb + x] - a[ra + x];
        s += d < 0 ? -d : d; c++;
      }
    }
    return c ? s / c : 1e9;
  }

  // Integer translation (dx,dy) such that prev[x+dx, y+dy] ≈ cur[x, y].
  _estimateShift(prevFull, curFull, prevL4, curL4, prevL16, curL16) {
    const w = this._w, h = this._h;
    const search = (a, b, lw, lh, cx, cy, rad, step, margin) => {
      let best = Infinity, bx = cx, by = cy;
      for (let dy = cy - rad; dy <= cy + rad; dy++) for (let dx = cx - rad; dx <= cx + rad; dx++) {
        const v = MotionDetector._sad(a, b, lw, lh, dx, dy, step, margin);
        if (v < best) { best = v; bx = dx; by = dy; }
      }
      return [bx, by];
    };
    let [dx, dy] = search(prevL16, curL16, this._w16, this._h16, 0, 0, Math.ceil(this.maxShift / 16), 1, 1);
    [dx, dy] = search(prevL4, curL4, this._w4, this._h4, dx * 4, dy * 4, 2, this.shiftStep4, 2);
    // Full-res refine on a sparse grid: ±2 px around the 1/4-res estimate.
    [dx, dy] = search(prevFull, curFull, w, h, dx * 4, dy * 4, 2, this.shiftStepFull, 4);
    return { dx, dy };
  }

  // Median |gx|+|gy| of the aligned frame two steps back, in a margin ring
  // around the blob box. Two frames back so the object's own ghost (one frame
  // back, often inside the ring) does not count; median so a single scenery
  // edge touching the ring does not either.
  _bgTexture(prev, off, b) {
    const w = this._w, h = this._h, hist = this._texHist;
    hist.fill(0);
    const m = Math.max(4, Math.min(16, Math.max(b.w, b.h)));
    const x0 = Math.max(1, b.x - m), x1 = Math.min(w - 2, b.x + b.w + m);
    const y0 = Math.max(1, b.y - m), y1 = Math.min(h - 2, b.y + b.h + m);
    const bx0 = b.x, bx1 = b.x + b.w, by0 = b.y, by1 = b.y + b.h;
    const lim = w * h - w - 1;
    let c = 0;
    for (let y = y0; y < y1; y += 2) {
      const inRow = y >= by0 && y < by1;
      for (let x = x0; x < x1; x += 2) {
        if (inRow && x >= bx0 && x < bx1) continue;
        const k = y * w + x + off;
        if (k < w + 1 || k >= lim) continue;
        let gx = prev[k + 1] - prev[k - 1]; if (gx < 0) gx = -gx;
        let gy = prev[k + w] - prev[k - w]; if (gy < 0) gy = -gy;
        let g = gx + gy; if (g > 255) g = 255;
        hist[g]++; c++;
      }
    }
    if (!c) return 0;
    const half = c >> 1;
    for (let v = 0, acc = 0; v < 256; v++) { acc += hist[v]; if (acc > half) return v; }
    return 255;
  }

  /**
   * @param {ImageData} imageData RGBA frame
   * @returns {Array<{x,y,w,h,area,label,confidence,source}>} blobs, largest first
   */
  update(imageData) {
    const { data, width: w, height: h } = imageData;
    this._ensureBuffers(w, h);
    const n = this._n;

    // Rotate ring: frames[0] = cur, [1] = prev1, [2] = prev2.
    const f = this._frames;
    const cur = f[2]; f[2] = f[1]; f[1] = f[0]; f[0] = cur;
    const prev1 = f[1], prev2 = f[2];
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      cur[i] = (data[j] * 77 + data[j + 1] * 150 + data[j + 2] * 29) >> 8;
    }
    // Pyramid for the current frame (swap with previous frame's pyramid).
    const p4 = this._p4, p16 = this._p16;
    const c4 = p4[1]; p4[1] = p4[0]; p4[0] = c4;
    const c16 = p16[1]; p16[1] = p16[0]; p16[0] = c16;
    MotionDetector._down4(cur, w, c4, this._w4, this._h4);
    MotionDetector._down4(c4, this._w4, c16, this._w16, this._h16);

    this._count++;
    if (this._count < 3) return [];     // need two previous frames

    // Shift estimation. s2 (prev2→prev1) is last frame's s1.
    this._s2.dx = this._s1.dx; this._s2.dy = this._s1.dy;
    if (this.stabilize) {
      const s = this._estimateShift(prev1, cur, p4[1], p4[0], p16[1], p16[0]);
      this._s1.dx = s.dx; this._s1.dy = s.dy;
    } else {
      this._s1.dx = this._s1.dy = 0;
    }
    this.lastShift = { dx: this._s1.dx, dy: this._s1.dy };
    if (Math.abs(this._s1.dx) > this.maxShift || Math.abs(this._s1.dy) > this.maxShift) return [];

    const dx1 = this._s1.dx, dy1 = this._s1.dy;
    const dx2 = dx1 + this._s2.dx, dy2 = dy1 + this._s2.dy;
    // Pixels whose aligned sources fall outside either previous frame are skipped.
    const bd = Math.max(1, this.border);
    const x0 = Math.max(bd, -dx1, -dx2), x1 = Math.min(w - bd, w - dx1, w - dx2);
    const y0 = Math.max(bd, -dy1, -dy2), y1 = Math.min(h - bd, h - dy1, h - dy2);
    const off1 = dy1 * w + dx1, off2 = dy2 * w + dx2;

    const t = this.threshold, eg = this.edgeGrad, ek = this.edgeK;
    const labels = this._labels;
    labels.fill(0);
    const cwc = this._cw, clutter = this._clutter, cellSum = this._cellSum, cellN = this._cellN, cellThr = this._cellThr;
    const kc = this.clutterK;
    for (let c = 0; c < cellThr.length; c++) { const v = kc * clutter[c]; cellThr[c] = v > t ? v | 0 : t; }
    cellSum.fill(0); cellN.fill(0);
    let parent = this._parent, area = this._area, sum = this._sum;
    let minX = this._minX, maxX = this._maxX, minY = this._minY, maxY = this._maxY;
    let next = 1, motionPixels = 0;

    for (let y = y0; y < y1; y++) {
      let i = y * w + x0;
      const cellRow = (y >> CELL_SHIFT) * cwc;
      for (let x = x0; x < x1; x++, i++) {
        const c = cur[i];
        let a = c - prev1[i + off1]; if (a < 0) a = -a;
        let b = c - prev2[i + off2]; if (b < 0) b = -b;
        const cell = cellRow + (x >> CELL_SHIFT);
        const dmin = a < b ? a : b;
        cellSum[cell] += dmin; cellN[cell]++;
        if (dmin <= cellThr[cell]) continue;

        // Static-edge test: strong gradient here in both cur and aligned prev1,
        // and the residual is no bigger than misalignment alone would explain.
        let gc = cur[i + 1] - cur[i - 1]; if (gc < 0) gc = -gc;
        let gcy = cur[i + w] - cur[i - w]; if (gcy < 0) gcy = -gcy;
        if (gc + gcy > eg) {
          const k = i + off1;
          let gp = prev1[k + 1] - prev1[k - 1]; if (gp < 0) gp = -gp;
          let gpy = prev1[k + w] - prev1[k - w]; if (gpy < 0) gpy = -gpy;
          const g = gp + gpy;
          if (g > eg) {
            const gs = gc + gcy < g ? gc + gcy : g;     // static edge strength
            const d = a < b ? a : b;
            if (d <= ek * gs) continue;
          }
        }

        motionPixels++;
        let la = labels[i - w], lb = labels[i - 1];
        let lbl;
        if (la && lb) {
          while (parent[la] !== la) { parent[la] = parent[parent[la]]; la = parent[la]; }
          while (parent[lb] !== lb) { parent[lb] = parent[parent[lb]]; lb = parent[lb]; }
          if (la < lb) { parent[lb] = la; lbl = la; } else { parent[la] = lb; lbl = lb; }
        } else if (la || lb) {
          lbl = la || lb;
        } else {
          if (next === this._cap) {
            this._allocLabels(this._cap * 2);
            parent = this._parent; area = this._area; sum = this._sum;
            minX = this._minX; maxX = this._maxX; minY = this._minY; maxY = this._maxY;
          }
          lbl = next++;
          parent[lbl] = lbl; area[lbl] = 0; sum[lbl] = 0;
          minX[lbl] = x; maxX[lbl] = x; minY[lbl] = y; maxY[lbl] = y;
        }
        labels[i] = lbl;
        area[lbl]++;
        sum[lbl] += a < b ? a : b;
        if (x < minX[lbl]) minX[lbl] = x; else if (x > maxX[lbl]) maxX[lbl] = x;
        if (y > maxY[lbl]) maxY[lbl] = y;
      }
    }

    // Update the per-cell clutter level from this frame's mean residual.
    const sm = this.clutterSmoothing;
    for (let c = 0; c < clutter.length; c++) {
      if (cellN[c]) clutter[c] += (cellSum[c] / cellN[c] - clutter[c]) * sm;
    }
    this.lastMotionFraction = motionPixels / n;
    if (motionPixels > n * this.maxMotionFraction) return [];

    for (let l = next - 1; l >= 1; l--) {
      let r = l;
      while (parent[r] !== r) r = parent[r];
      if (r === l) continue;
      area[r] += area[l];
      sum[r] += sum[l];
      if (minX[l] < minX[r]) minX[r] = minX[l];
      if (maxX[l] > maxX[r]) maxX[r] = maxX[l];
      if (minY[l] < minY[r]) minY[r] = minY[l];
      if (maxY[l] > maxY[r]) maxY[r] = maxY[l];
    }

    // Raw components → merge fragments → size/shape filters.
    const raw = [];
    for (let l = 1; l < next; l++) {
      if (parent[l] !== l) continue;
      const a = area[l];
      if (a < this.minArea) continue;
      raw.push({ x: minX[l], y: minY[l], x1: maxX[l] + 1, y1: maxY[l] + 1, area: a, sum: sum[l] });
    }
    raw.sort((p, q) => q.area - p.area);
    if (raw.length > 400) raw.length = 400;
    const gap = this.mergeGap;
    const merged = [];
    for (const r of raw) {
      let target = null;
      for (const m of merged) {
        if (r.x < m.x1 + gap && m.x < r.x1 + gap && r.y < m.y1 + gap && m.y < r.y1 + gap) { target = m; break; }
      }
      if (target) {
        if (r.x < target.x) target.x = r.x; if (r.y < target.y) target.y = r.y;
        if (r.x1 > target.x1) target.x1 = r.x1; if (r.y1 > target.y1) target.y1 = r.y1;
        target.area += r.area; target.sum += r.sum;
      } else {
        merged.push({ x: r.x, y: r.y, x1: r.x1, y1: r.y1, area: r.area, sum: r.sum });
      }
    }
    // One more pass so chains of fragments that only became adjacent after a
    // merge are joined too.
    for (let i = 0; i < merged.length; i++) {
      const m = merged[i];
      if (!m) continue;
      for (let j = i + 1; j < merged.length; j++) {
        const o = merged[j];
        if (!o) continue;
        if (o.x < m.x1 + gap && m.x < o.x1 + gap && o.y < m.y1 + gap && m.y < o.y1 + gap) {
          if (o.x < m.x) m.x = o.x; if (o.y < m.y) m.y = o.y;
          if (o.x1 > m.x1) m.x1 = o.x1; if (o.y1 > m.y1) m.y1 = o.y1;
          m.area += o.area; m.sum += o.sum; merged[j] = null;
        }
      }
    }

    const cw = this._cw, ch = this._ch;
    const cellHits = this._cellHits, cellNow = this._cellNow, activity = this._activity;
    cellHits.fill(0); cellNow.fill(0);
    const cand = [];
    for (const m of merged) {
      if (!m) continue;
      const a = m.area;
      if (a > this.maxArea) continue;
      const bw = m.x1 - m.x, bh = m.y1 - m.y;
      if (bw > this.maxSide || bh > this.maxSide) continue;
      const contrast = m.sum / a;
      if (bw > this.bigSide || bh > this.bigSide) {
        if (contrast < this.bigContrast) continue;
      } else if (a < this.minFill * bw * bh) continue;
      const cx = ((m.x + m.x1) >> 1) / CELL | 0, cy = ((m.y + m.y1) >> 1) / CELL | 0;
      cellHits[cy * cw + cx]++;
      cellNow[cy * cw + cx] = 1;
      cand.push({ x: m.x, y: m.y, w: bw, h: bh, area: a, contrast, cell: cy * cw + cx });
    }

    // Activity EMA: cells that produced a blob this frame move toward 1, others decay.
    const decay = this.activityDecay;
    for (let c = 0; c < activity.length; c++) activity[c] = activity[c] * decay + (cellNow[c] ? 1 - decay : 0);

    if (cand.length > this.maxBlobsFrame) return [];

    const blobs = [];
    const clusterMax = this.clusterMax, actMax = this.activityMax;
    for (const b of cand) {
      const c = b.cell, cx = c % cw, cy = (c / cw) | 0;
      let hits = 0;
      for (let yy = Math.max(0, cy - 1); yy <= Math.min(ch - 1, cy + 1); yy++)
        for (let xx = Math.max(0, cx - 1); xx <= Math.min(cw - 1, cx + 1); xx++) hits += cellHits[yy * cw + xx];
      if (hits > clusterMax) continue;
      if (activity[c] > actMax) continue;
      // Background texture must be low in absolute terms, or small next to
      // the blob's own contrast (a dark drone over soft cloud texture).
      const tex = this._bgTexture(prev2, off2, b);
      if (tex > this.bgTexMax && tex > b.contrast * this.bgTexRatio) continue;
      blobs.push({
        x: b.x, y: b.y, w: b.w, h: b.h,
        area: b.area, contrast: b.contrast, label: '', confidence: -1, source: 'motion',
      });
    }
    blobs.sort((p, q) => q.area - p.area);
    if (blobs.length > this.maxBlobs) blobs.length = this.maxBlobs;
    return blobs;
  }
}
