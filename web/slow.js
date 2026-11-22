// slow.js — Long-temporal-baseline detector for slow movers (a fly walking on
// glass, a hovering insect, a drifting drone far away).
//
// motion.js compares each frame with the one before it, so it only sees things
// that move about their own size per frame. Here the current frame is compared
// with the frames K and 2K back instead (K = `slowBaseline`, 8 frames): an
// 8x12 px fly walking 0.7 px/frame has moved half its length after K frames
// and its whole length after 2K.
//
// Everything runs on a half-resolution grey image (2x2 box average): a quarter
// of the pixels, half the sensor / compression noise, and slow objects do not
// need single-pixel timing.
//
// Per frame, no allocations in steady state except the small result arrays:
//   1. Alignment of the two old frames to the current one, estimated DIRECTLY
//      between the frame pairs (SAD search around the accumulated per-frame
//      shifts). Summing rounded per-frame shifts instead loses a slow drift
//      completely: 0.4 px/frame rounds to 0 every frame, 6 px after 16 frames.
//   2. Exposure compensation: one gain per old frame (ratio of mean grey),
//      applied through a lookup table. Phone cameras re-expose by several
//      percent within half a second, which on bright sky is the same size as
//      the detection threshold.
//   3. Residual = min(|cur - old_K|, |cur - old_2K|), both with the same sign
//      (an object is darker, or brighter, than what was there before in both
//      old frames). The min removes the ghost at the old position.
//   4. Misalignment tolerance: a residual pixel only counts if the current
//      value is outside the value range of the 3x3 neighbourhood in both old
//      frames (plus threshold). Static dirt and scenery edges that are off by
//      up to one half-res pixel (sub-pixel shift, slight rotation) vanish.
//   5. Connected components, merge, and per-blob tests:
//        - size / fill ratio (compact objects only),
//        - "present now": the blob must stand out from its surroundings in the
//          current frame and not in the old one. Rejects the hole a resting
//          object leaves when it takes off (residual for K frames at a fixed
//          place, i.e. exactly what a slow object would look like),
//        - smooth background around it (same texture rule as motion.js),
//        - not one of many candidates in the same neighbourhood (foliage).
//   6. Slow tracks. A candidate must reappear at (nearly) the same scene
//      position on several passes before it is reported. A reported object
//      that stops moving produces no residual any more; its mark is held as
//      long as the pixels that changed when it arrived still look like the
//      object and not like the background that was there before (two stored
//      patches per track). Static dirt can never get a track, because a track
//      is only ever born from motion.

const CELL_SHIFT = 4;                 // clutter cell: 16 half-res px = 32 full-res px, as in motion.js
const TPL = 16;                       // appearance patch side in half-res px (32 px full-res: a fly plus margin)
const TPL_N = TPL * TPL;
const INITIAL_LABELS = 1024;

function makeTrack() {
  return {
    active: false, id: 0,
    x: 0, y: 0, w: 0, h: 0,           // half-res centre and box size
    hits: 0, age: 0, missed: 0, held: 0,
    confirmed: false, pol: 0, contrast: 0, area: 0,
    qx0: 0, qy0: 0, travel: 0,        // scene position at birth, largest distance from it
    hasTpl: false, sig: 0, maskN: 0,
    tpl: new Uint8Array(TPL_N),       // object appearance at the last residual match
    bg: new Uint8Array(TPL_N),        // what was there before the object arrived
    mask: new Uint8Array(TPL_N),      // pixels where the two differ
    matched: false,
  };
}

export class SlowMoverDetector {
  constructor(options = {}) {
    this.threshold = options.threshold ?? 20;
    this.baseline = options.slowBaseline ?? 8;          // K, frames
    this.tolerance = options.slowTolerance ?? 1;        // neighbourhood radius, half-res px
    this.minArea = options.slowMinArea ?? 2;            // half-res px (= 8 full-res px)
    this.maxSide = options.slowMaxSide ?? 40;           // half-res px; larger slow changes are scenery (clouds, shadows, parallax)
    this.minFill = options.slowMinFill ?? 0.3;
    this.mergeGap = options.slowMergeGap ?? 3;          // half-res px
    // Per-cell adaptive threshold as in motion.js: max(threshold, clutterK ×
    // the cell's recent mean residual). The mean here only counts residuals
    // above half the threshold and leaves out accepted candidates, so a slow
    // object does not raise the threshold of the cell it lives in.
    this.clutterK = options.slowClutterK ?? 12;
    this.clutterSmoothing = options.clutterSmoothing ?? 0.25;
    this.bgTexMax = options.slowTexMax ?? 6;
    this.bgTexRatio = options.bgTexRatio ?? 0.3;
    this.clusterMax = options.slowClusterMax ?? 2;
    this.maxCandidates = options.slowMaxCandidates ?? 12;   // more than this per pass = alignment failed / camera moving
    this.maxRaw = options.slowMaxRaw ?? 12;                 // residual components per pass; more = the camera moves, not insects
    this.minTravel = options.slowMinTravel ?? 3;            // half-res px a candidate must move from where it was first seen
    this.minHits = options.slowMinHits ?? 3;
    this.maxMissed = options.slowMaxMissed ?? 2;
    this.matchDist = options.slowMatchDist ?? 6;        // half-res px
    this.presence = options.slowPresence ?? 0.5;
    // A resting object keeps its mark for holdPerHit frames per frame it was
    // seen moving (at least holdFloor, at most holdMax). A fly that walked for
    // a second may rest for two; a cloud edge or the hole a departing drone
    // leaves, seen for a handful of frames, is let go after a quarter second.
    this.holdFloor = options.slowHoldFloor ?? 8;
    this.holdPerHit = options.slowHoldPerHit ?? 2;
    this.holdMax = options.slowHoldMax ?? 300;
    this.maxTracks = options.slowMaxTracks ?? 24;
    this.border = options.slowBorder ?? 4;
    this.sadStep = options.slowSadStep ?? 8;
    this.posGain = options.slowPosGain ?? 0.5;                // sampling stride of the alignment SAD, half-res px
    // Local-alignment veto (see _localExplains): annulus from localInner px
    // beyond the blob to localOuter px from its centre, half-res.
    this.localInner = options.slowLocalInner ?? 8;         // 2 px/frame × K frames = 16 px = 8 half-res px
    this.localOuter = options.slowLocalOuter ?? 32;
    this.localGain = options.slowLocalGain ?? 0.8;
    this.localRad = options.slowLocalRad ?? 3;             // minimum local search radius, half-res px
    this._acc = new Float64Array(11 * 11);
    this.debug = options.slowDebug ? { cands: [], rej: [], info: {} } : null;

    this._w = 0; this._h = 0;
    this._count = 0;
    this._tracks = [];
    for (let i = 0; i < this.maxTracks; i++) this._tracks.push(makeTrack());
    this._nextId = 1;
    this._lutA = new Uint8Array(256); this._lutB = new Uint8Array(256);
    this._hist = new Uint32Array(256);
    this._allocLabels(INITIAL_LABELS);
  }

  setThreshold(v) { this.threshold = v; }

  reset() {
    this._count = 0;
    for (const t of this._tracks) t.active = false;
    if (this._clutter) this._clutter.fill(0);
  }

  _allocLabels(cap) {
    const grow = (old, Type) => { const a = new Type(cap); if (old) a.set(old); return a; };
    this._parent = grow(this._parent, Int32Array);
    this._area = grow(this._area, Int32Array);
    this._sum = grow(this._sum, Int32Array);
    this._pol = grow(this._pol, Int32Array);
    this._minX = grow(this._minX, Int32Array);
    this._maxX = grow(this._maxX, Int32Array);
    this._minY = grow(this._minY, Int32Array);
    this._maxY = grow(this._maxY, Int32Array);
    this._cap = cap;
  }

  _ensure(w, h) {
    if (this._w === w && this._h === h) return;
    this._w = w; this._h = h;
    const hw = this._hw = w >> 1, hh = this._hh = h >> 1;
    const hn = this._hn = hw * hh;
    this._N = 2 * this.baseline + 1;
    this._ring = [];
    for (let i = 0; i < this._N; i++) this._ring.push(new Uint8Array(hn));
    this._rx = new Float64Array(this._N);      // mosaic position of each stored frame, half-res px
    this._ry = new Float64Array(this._N);
    this._head = -1;
    this._labels = new Int32Array(hn);
    this._list = new Int32Array(hn >> 4);      // pixels over threshold; more than 1/16 of the frame = unusable pass
    const cw = this._cw = Math.ceil(hw / (1 << CELL_SHIFT)), ch = this._ch = Math.ceil(hh / (1 << CELL_SHIFT));
    this._clutter = new Float32Array(cw * ch);
    this._cellSum = new Uint32Array(cw * ch);
    this._cellThr = new Int32Array(cw * ch);
    this._cellHits = new Uint16Array(cw * ch);
    this._count = 0;
    for (const t of this._tracks) t.active = false;
  }

  // 2x2 box average.
  static _half(src, w, dst, hw, hh) {
    for (let y = 0, o = 0; y < hh; y++) {
      let i = 2 * y * w;
      for (let x = 0; x < hw; x++, i += 2) dst[o++] = (src[i] + src[i + 1] + src[i + w] + src[i + w + 1] + 2) >> 2;
    }
  }

  // Mean |b[x,y] - a[x+dx,y+dy]| on a sparse grid.
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

  // Integer (dx,dy) near the guess minimising the SAD of old→cur: steepest
  // descent over the 8 neighbours, at most `rad` steps. The guess (previous
  // refined position + this frame's shift) is within a pixel in steady state,
  // so this usually costs 9 SAD evaluations instead of (2·rad+1)². A move
  // must improve the SAD by 2 %, so featureless frames do not wander.
  _refine(old, cur, gx, gy, rad, out) {
    const hw = this._hw, hh = this._hh, m = 8, step = this.sadStep;
    let bx = gx, by = gy, best = SlowMoverDetector._sad(old, cur, hw, hh, gx, gy, step, m);
    for (let it = 0; it < rad; it++) {
      let nx = bx, ny = by, nb = best * 0.98;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const v = SlowMoverDetector._sad(old, cur, hw, hh, bx + dx, by + dy, step, m);
        if (v < nb) { nb = v; nx = bx + dx; ny = by + dy; }
      }
      if (nx === bx && ny === by) break;
      bx = nx; by = ny; best = nb;
    }
    out.dx = bx; out.dy = by;
  }

  // Lookup table mapping old grey values to the current exposure.
  _gainLut(old, cur, dx, dy, lut) {
    const w = this._hw, h = this._hh, m = 6, step = 4;
    const x0 = Math.max(m, m - dx), x1 = Math.min(w - m, w - m - dx);
    const y0 = Math.max(m, m - dy), y1 = Math.min(h - m, h - m - dy);
    let sc = 0, so = 0;
    for (let y = y0; y < y1; y += step) {
      const rb = y * w, ra = (y + dy) * w + dx;
      for (let x = x0; x < x1; x += step) { sc += cur[rb + x]; so += old[ra + x]; }
    }
    let g = so > 0 ? sc / so : 1;
    if (g < 0.75) g = 0.75; else if (g > 1.33) g = 1.33;
    for (let v = 0; v < 256; v++) { const q = v * g + 0.5; lut[v] = q > 255 ? 255 : q | 0; }
    return g;
  }

  // Residual scan over the valid rectangle. Returns the number of pixels whose
  // residual exceeds their cell's threshold with the same sign in both old
  // frames; their indices go to this._list (count may exceed its length).
  // Nearly all pixels leave at the first comparison (unchanged against frame
  // A), which is what keeps this pass cheap. The per-cell clutter sum counts
  // residuals above half the threshold only (sensor noise is not clutter).
  _scan(cur, A, B, offA, offB, x0, x1, y0, y1) {
    const hw = this._hw, cw = this._cw, lutA = this._lutA, lutB = this._lutB;
    const cellThr = this._cellThr, cellSum = this._cellSum, list = this._list, cap = list.length;
    const half = this.threshold >> 1;
    let n = 0;
    for (let y = y0; y < y1; y++) {
      const cellRow = (y >> CELL_SHIFT) * cw;
      let x = x0, i = y * hw + x0;
      while (x < x1) {
        const cell = cellRow + (x >> CELL_SHIFT);
        let xe = ((x >> CELL_SHIFT) + 1) << CELL_SHIFT; if (xe > x1) xe = x1;
        const thr = cellThr[cell];
        const ie = i + (xe - x);
        let s = 0;
        for (; i < ie; i++) {
          const c = cur[i];
          const a = c - lutA[A[i + offA]];
          const aa = a < 0 ? -a : a;
          if (aa <= half) continue;
          const b = c - lutB[B[i + offB]];
          const ab = b < 0 ? -b : b;
          if (ab <= half) continue;
          const m = aa < ab ? aa : ab;
          s += m;
          if (m > thr && (a ^ b) >= 0) { if (n < cap) list[n] = i; n++; }
        }
        cellSum[cell] += s;
        x = xe;
      }
    }
    return n;
  }

  // Does the scenery around blob m line up better at some other local shift
  // of old frame A, and does that shift make the blob vanish? Then the blob is
  // scenery the global translation does not describe at that place: parallax
  // between the window and what is behind it, rotation, zoom, rolling
  // shutter, a swaying branch. "Around" is a square annulus: the blob itself
  // and everything within reach of where it could have been K frames ago is
  // left out, so a lone fly on plain sky cannot explain itself away.
  _localExplains(cur, A, lutA, ax, ay, m, rad) {
    const hw = this._hw, hh = this._hh, labels = this._labels, acc = this._acc;
    const side = 2 * rad + 1, nShift = side * side;
    const cx = (m.x + m.x1) >> 1, cy = (m.y + m.y1) >> 1;
    const ex = (Math.max(m.w, m.h) >> 1) + this.localInner + rad;
    const W = this.localOuter, pad = rad + 2;
    const x0 = Math.max(cx - W, pad, pad - ax), x1 = Math.min(cx + W, hw - pad, hw - pad - ax);
    const y0 = Math.max(cy - W, pad, pad - ay), y1 = Math.min(cy + W, hh - pad, hh - pad - ay);
    const offA = ay * hw + ax;
    for (let k = 0; k < nShift; k++) acc[k] = 0;
    let n = 0;
    for (let y = y0; y < y1; y += 2) {
      const inRow = y >= cy - ex && y <= cy + ex;
      for (let x = x0; x < x1; x += 2) {
        if (inRow && x >= cx - ex && x <= cx + ex) continue;
        const i = y * hw + x, c = cur[i];
        let k = 0;
        for (let v = -rad; v <= rad; v++) {
          const row = i + offA + v * hw;
          for (let u = -rad; u <= rad; u++, k++) { const d = c - lutA[A[row + u]]; acc[k] += d < 0 ? -d : d; }
        }
        n++;
      }
    }
    if (n < 64) return false;
    const centre = rad * side + rad;
    let best = acc[centre], bk = centre;
    for (let k = 0; k < nShift; k++) if (acc[k] < best) { best = acc[k]; bk = k; }
    if (bk === centre || best > this.localGain * acc[centre]) return false;   // no clearly better local alignment
    const bu = bk % side - rad, bv = ((bk / side) | 0) - rad;
    // Blob residual against A at the local shift (and its 8 neighbours: sub-pixel).
    let low = Infinity;
    for (let v = bv - 1; v <= bv + 1; v++) for (let u = bu - 1; u <= bu + 1; u++) {
      let sres = 0, cnt = 0;
      for (let y = m.y; y < m.y1; y++) for (let x = m.x, i = y * hw + m.x; x < m.x1; x++, i++) {
        if (!labels[i]) continue;
        const j = i + offA + v * hw + u;
        const d = cur[i] - lutA[A[j]];
        sres += d < 0 ? -d : d; cnt++;
      }
      if (cnt && sres / cnt < low) low = sres / cnt;
    }
    m.localShift = [bu, bv]; m.localRes = low;
    return low < 0.5 * m.contrast;
  }

  // True if the value range of the (2r+1)² neighbourhood of old[k] stays on
  // the far side of `lim`: all above it (dark object) or all below (bright).
  _nbClear(old, lut, k, lim, dark) {
    const hw = this._hw, r = this.tolerance;
    for (let yy = -r; yy <= r; yy++) {
      const row = k + yy * hw;
      for (let xx = -r; xx <= r; xx++) {
        const v = lut[old[row + xx]];
        if (dark ? v <= lim : v >= lim) return false;
      }
    }
    return true;
  }

  // Median |gx|+|gy| of `img` (offset `off`) in a ring around the box.
  _ringTexture(img, off, x0b, y0b, x1b, y1b) {
    const w = this._hw, h = this._hh, hist = this._hist;
    hist.fill(0);
    const m = Math.max(3, Math.min(8, Math.max(x1b - x0b, y1b - y0b)));
    const x0 = Math.max(1, x0b - m), x1 = Math.min(w - 1, x1b + m);
    const y0 = Math.max(1, y0b - m), y1 = Math.min(h - 1, y1b + m);
    const lim = w * h - w - 1;
    let c = 0;
    for (let y = y0; y < y1; y++) {
      const inRow = y >= y0b && y < y1b;
      for (let x = x0; x < x1; x++) {
        if (inRow && x >= x0b && x < x1b) continue;
        const k = y * w + x + off;
        if (k < w + 1 || k >= lim) continue;
        let gx = img[k + 1] - img[k - 1]; if (gx < 0) gx = -gx;
        let gy = img[k + w] - img[k - w]; if (gy < 0) gy = -gy;
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
   * @param {Uint8Array} gray  full-resolution grey frame
   * @param {{dx:number,dy:number}} shift  camera shift previous → current frame (full-res px)
   * @param {boolean} cut      true if the shift estimate is unusable (scene cut / violent shake)
   * @returns {Array} blobs in full-res pixels, one per confirmed slow object
   */
  update(gray, w, h, shift, cut) {
    this._ensure(w, h);
    const hw = this._hw, hh = this._hh, N = this._N;
    if (cut) this.reset();
    const prevHead = this._head;
    const head = this._head = (this._head + 1) % N;
    const cur = this._ring[head];
    SlowMoverDetector._half(gray, w, cur, hw, hh);
    this._count++;
    const dbg = this.debug;
    if (dbg) { dbg.cands = []; dbg.rej = []; dbg.info = {}; }

    // Mosaic position of this frame (static scenery has constant image
    // position + mosaic position). Short term it follows the per-frame shift;
    // the direct measurement against frame A pulls it back by `posGain`, so
    // sub-pixel drift the per-frame estimate cannot see does not accumulate,
    // while single-pixel rounding of one measurement does not make it jump.
    let rx = 0, ry = 0, prx = 0, pry = 0;
    if (this._count > 1) { prx = this._rx[prevHead]; pry = this._ry[prevHead]; rx = prx + shift.dx / 2; ry = pry + shift.dy / 2; }

    const K = Math.min(this.baseline, (this._count - 1) >> 1);
    let cands = null;
    if (K >= 2) {
      const ia = (head - K + N) % N, ib = (head - 2 * K + N) % N;
      const A = this._ring[ia], B = this._ring[ib];
      const o = this._o || (this._o = { dx: 0, dy: 0 });
      this._refine(A, cur, Math.round(rx - this._rx[ia]), Math.round(ry - this._ry[ia]), 2, o);
      const ax = o.dx, ay = o.dy;
      const g = this.posGain;
      const ex = this._rx[ia] + ax - rx, ey = this._ry[ia] + ay - ry;
      // Within ±0.5 px the measurement (an integer) carries no information.
      if (ex > 0.5) rx += g * (ex - 0.5); else if (ex < -0.5) rx += g * (ex + 0.5);
      if (ey > 0.5) ry += g * (ey - 0.5); else if (ey < -0.5) ry += g * (ey + 0.5);
      this._refine(B, cur, Math.round(rx - this._rx[ib]), Math.round(ry - this._ry[ib]), 2, o);
      const bx = o.dx, by = o.dy;
      cands = this._residual(cur, A, B, ax, ay, bx, by);
    }
    this._rx[head] = rx; this._ry[head] = ry;

    return this._track(cur, cands, rx - prx, ry - pry, rx, ry);
  }

  // Steps 2-5: candidates of this pass, in half-res pixels. null = unusable pass.
  _residual(cur, A, B, ax, ay, bx, by) {
    const hw = this._hw, hh = this._hh, r = this.tolerance, dbg = this.debug;
    const lutA = this._lutA, lutB = this._lutB;
    this._gainLut(A, cur, ax, ay, lutA);
    this._gainLut(B, cur, bx, by, lutB);
    const offA = ay * hw + ax, offB = by * hw + bx;
    const bd = Math.max(this.border, r + 1);
    const x0 = Math.max(bd, r + 1 - ax, r + 1 - bx), x1 = Math.min(hw - bd, hw - r - 1 - ax, hw - r - 1 - bx);
    const y0 = Math.max(bd, r + 1 - ay, r + 1 - by), y1 = Math.min(hh - bd, hh - r - 1 - ay, hh - r - 1 - by);

    const t = this.threshold, kc = this.clutterK;
    const cw = this._cw, clutter = this._clutter, cellSum = this._cellSum, cellThr = this._cellThr;
    for (let c = 0; c < cellThr.length; c++) { const v = kc * clutter[c]; cellThr[c] = v > t ? v | 0 : t; }
    cellSum.fill(0);
    const labels = this._labels;
    labels.fill(0);
    let parent = this._parent, area = this._area, sum = this._sum, pol = this._pol;
    let minX = this._minX, maxX = this._maxX, minY = this._minY, maxY = this._maxY;
    let next = 1;

    // Stage 1 (every pixel, kept tight): min-of-two residual, per-cell
    // clutter sums, indices of pixels over threshold.
    const list = this._list;
    const nList = this._scan(cur, A, B, offA, offB, x0, x1, y0, y1);
    if (dbg) dbg.info.pix = nList;
    // Stage 2 (the few pixels over threshold, in raster order): tolerance
    // test and connected components.
    const nOk = nList > list.length ? 0 : nList;
    for (let k = 0; k < nOk; k++) {
      const i = list[k];
      const y = (i / hw) | 0, x = i - y * hw;
      const c = cur[i];
      const a = c - lutA[A[i + offA]], b = c - lutB[B[i + offB]];
      const aa = a < 0 ? -a : a, ab = b < 0 ? -b : b;
      const m = aa < ab ? aa : ab;
      const thr = cellThr[(y >> CELL_SHIFT) * cw + (x >> CELL_SHIFT)];
      const dark = a < 0;
      const lim = dark ? c + thr : c - thr;
      if (!this._nbClear(A, lutA, i + offA, lim, dark) || !this._nbClear(B, lutB, i + offB, lim, dark)) continue;

      let la = labels[i - hw], lb = labels[i - 1];
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
          parent = this._parent; area = this._area; sum = this._sum; pol = this._pol;
          minX = this._minX; maxX = this._maxX; minY = this._minY; maxY = this._maxY;
        }
        lbl = next++;
        parent[lbl] = lbl; area[lbl] = 0; sum[lbl] = 0; pol[lbl] = 0;
        minX[lbl] = x; maxX[lbl] = x; minY[lbl] = y; maxY[lbl] = y;
      }
      labels[i] = lbl;
      area[lbl]++;
      sum[lbl] += m;
      pol[lbl] += dark ? -1 : 1;
      if (x < minX[lbl]) minX[lbl] = x; else if (x > maxX[lbl]) maxX[lbl] = x;
      if (y > maxY[lbl]) maxY[lbl] = y;
    }

    for (let l = next - 1; l >= 1; l--) {
      let q = l;
      while (parent[q] !== q) q = parent[q];
      if (q === l) continue;
      area[q] += area[l]; sum[q] += sum[l]; pol[q] += pol[l];
      if (minX[l] < minX[q]) minX[q] = minX[l];
      if (maxX[l] > maxX[q]) maxX[q] = maxX[l];
      if (minY[l] < minY[q]) minY[q] = minY[l];
      if (maxY[l] > maxY[q]) maxY[q] = maxY[l];
    }
    const raw = [];
    for (let l = 1; l < next; l++) {
      if (parent[l] !== l) continue;
      raw.push({ x: minX[l], y: minY[l], x1: maxX[l] + 1, y1: maxY[l] + 1, area: area[l], sum: sum[l], pol: pol[l] });
    }
    if (dbg) dbg.info.raw = raw.length;
    if (nList > list.length || raw.length > this.maxRaw) { this._updateClutter(null); return null; }
    raw.sort((p, q) => q.area - p.area);
    const gap = this.mergeGap;
    const merged = [];
    for (const q of raw) {
      let target = null;
      for (const m of merged) {
        if (q.x < m.x1 + gap && m.x < q.x1 + gap && q.y < m.y1 + gap && m.y < q.y1 + gap) { target = m; break; }
      }
      if (target) {
        if (q.x < target.x) target.x = q.x; if (q.y < target.y) target.y = q.y;
        if (q.x1 > target.x1) target.x1 = q.x1; if (q.y1 > target.y1) target.y1 = q.y1;
        target.area += q.area; target.sum += q.sum; target.pol += q.pol;
      } else merged.push(q);
    }

    const ch = this._ch, cellHits = this._cellHits;
    cellHits.fill(0);
    const pre = [];
    for (const m of merged) {
      const bw = m.x1 - m.x, bh = m.y1 - m.y;
      m.w = bw; m.h = bh; m.contrast = m.sum / m.area;
      m.cell = ((m.y + m.y1) >> 1 >> CELL_SHIFT) * cw + ((m.x + m.x1) >> 1 >> CELL_SHIFT);
      cellHits[m.cell]++;
      if (m.area < this.minArea) { if (dbg) dbg.rej.push({ m, why: 'minArea' }); continue; }
      if (bw > this.maxSide || bh > this.maxSide) { if (dbg) dbg.rej.push({ m, why: 'maxSide' }); continue; }
      if (m.area < this.minFill * bw * bh) { if (dbg) dbg.rej.push({ m, why: 'minFill' }); continue; }
      pre.push(m);
    }
    const out = [];
    for (const m of pre) {
      const cx = m.cell % cw, cy = (m.cell / cw) | 0;
      let hits = 0;
      for (let yy = Math.max(0, cy - 1); yy <= Math.min(ch - 1, cy + 1); yy++)
        for (let xx = Math.max(0, cx - 1); xx <= Math.min(cw - 1, cx + 1); xx++) hits += cellHits[yy * cw + xx];
      if (hits > this.clusterMax) { if (dbg) dbg.rej.push({ m, why: 'cluster' }); continue; }

      // Present now: mean of the residual pixels against the ring around the
      // box, in the current frame and in old frame A.
      const g = 3;
      const xa = Math.max(0, m.x - g), xb = Math.min(hw, m.x1 + g), ya = Math.max(0, m.y - g), yb = Math.min(hh, m.y1 + g);
      let sC = 0, sA = 0, nM = 0, rC = 0, rA = 0, nR = 0;
      for (let y = ya; y < yb; y++) {
        const inRow = y >= m.y && y < m.y1;
        for (let x = xa, i = y * hw + xa; x < xb; x++, i++) {
          if (inRow && x >= m.x && x < m.x1) {
            if (labels[i]) { sC += cur[i]; sA += lutA[A[i + offA]]; nM++; }
          } else if (!labels[i]) { rC += cur[i]; rA += lutA[A[i + offA]]; nR++; }
        }
      }
      if (!nM || !nR) continue;
      const sign = m.pol < 0 ? -1 : 1;
      const stC = sign * (sC / nM - rC / nR), stA = sign * (sA / nM - rA / nR);
      m.stand = stC; m.standOld = stA;
      if (stC < 0.5 * this.threshold || stC < 2 * stA) { if (dbg) dbg.rej.push({ m, why: 'notPresent' }); continue; }

      const tex = this._ringTexture(B, offB, m.x, m.y, m.x1, m.y1);
      m.tex = tex;
      if (tex > this.bgTexMax && tex > m.contrast * this.bgTexRatio) { if (dbg) dbg.rej.push({ m, why: 'texture' }); continue; }
      // The faster the camera moved over the baseline, the larger the
      // parallax between scene layers can be.
      const rad = Math.min(5, this.localRad + Math.ceil(0.25 * Math.hypot(ax, ay)));
      if (this._localExplains(cur, A, lutA, ax, ay, m, rad)) { if (dbg) dbg.rej.push({ m, why: 'local' }); continue; }
      out.push(m);
    }
    if (dbg) { dbg.info.cands = out.length; dbg.cands = out; }
    if (out.length > this.maxCandidates) { this._updateClutter(null); return null; }
    this._updateClutter(out);
    this._offB = offB;
    return out;
  }

  _updateClutter(accepted) {
    const clutter = this._clutter, cellSum = this._cellSum, sm = this.clutterSmoothing;
    const cellPix = (1 << CELL_SHIFT) * (1 << CELL_SHIFT);
    if (accepted) for (const m of accepted) cellSum[m.cell] = cellSum[m.cell] > m.sum ? cellSum[m.cell] - m.sum : 0;
    for (let c = 0; c < clutter.length; c++) clutter[c] += (cellSum[c] / cellPix - clutter[c]) * sm;
  }

  // Store the appearance patches of track t at its current position.
  _snap(t, cur, B, offB, lutB) {
    const hw = this._hw, hh = this._hh;
    const cx = Math.round(t.x), cy = Math.round(t.y), x0 = cx - (TPL >> 1), y0 = cy - (TPL >> 1);
    t.hasTpl = false;
    if (x0 < 1 || y0 < 1 || x0 + TPL >= hw - 1 || y0 + TPL >= hh - 1) return;
    const lim = hw * hh;
    const half = this.threshold >> 1;
    const mx0 = Math.round(t.x - t.w / 2) - 2 - x0, mx1 = Math.round(t.x + t.w / 2) + 2 - x0;
    const my0 = Math.round(t.y - t.h / 2) - 2 - y0, my1 = Math.round(t.y + t.h / 2) + 2 - y0;
    let sig = 0, n = 0;
    for (let y = 0, k = 0; y < TPL; y++) {
      const row = (y0 + y) * hw + x0;
      for (let x = 0; x < TPL; x++, k++) {
        const c = cur[row + x];
        const j = row + x + offB;
        const b = j >= 0 && j < lim ? lutB[B[j]] : c;
        t.tpl[k] = c; t.bg[k] = b;
        let d = c - b; if (d < 0) d = -d;
        const on = d > half && x >= mx0 && x < mx1 && y >= my0 && y < my1;
        t.mask[k] = on ? 1 : 0;
        if (on) { sig += d; n++; }
      }
    }
    t.sig = sig; t.maskN = n;
    t.hasTpl = n >= 2;
  }

  // Is the object of track t still where it was last seen (±1 px)? Moves the
  // track to the best-fitting offset.
  _stillThere(t, cur) {
    if (!t.hasTpl) return false;
    const hw = this._hw, hh = this._hh;
    const cx = Math.round(t.x), cy = Math.round(t.y), x0 = cx - (TPL >> 1), y0 = cy - (TPL >> 1);
    if (x0 < 2 || y0 < 2 || x0 + TPL >= hw - 2 || y0 + TPL >= hh - 2) return false;
    let best = Infinity, bu = 0, bv = 0;
    for (let v = -1; v <= 1; v++) for (let u = -1; u <= 1; u++) {
      let s = 0;
      for (let y = 0, k = 0; y < TPL; y++) {
        const row = (y0 + y + v) * hw + x0 + u;
        for (let x = 0; x < TPL; x++, k++) {
          if (!t.mask[k]) continue;
          const d = cur[row + x] - t.tpl[k];
          s += d < 0 ? -d : d;
        }
      }
      if (u === 0 && v === 0) s *= 0.9;          // prefer staying put
      if (s < best) { best = s; bu = u; bv = v; }
    }
    if (best > this.presence * t.sig) return false;
    // Still an object: the masked pixels must stand out from the rest of the
    // patch in the current frame, with the polarity the object arrived with.
    // The patch of a departed object (sky where the drone was) matches
    // itself but stands out from nothing.
    let sm = 0, nm = 0, sr = 0, nr = 0;
    for (let y = 0, k = 0; y < TPL; y++) {
      const row = (y0 + y + bv) * hw + x0 + bu;
      for (let x = 0; x < TPL; x++, k++) { if (t.mask[k]) { sm += cur[row + x]; nm++; } else { sr += cur[row + x]; nr++; } }
    }
    if (!nm || !nr || t.pol * (sm / nm - sr / nr) < 0.5 * this.threshold) return false;
    t.x += bu; t.y += bv;
    return true;
  }

  // Step 6. (mx,my) = camera motion of this frame in half-res px, (rx,ry) = mosaic position.
  _track(cur, cands, mx, my, rx, ry) {
    const tracks = this._tracks, dbg = this.debug;
    const head = this._head, N = this._N;
    // A static scene point moves by -shift in the image.
    for (const t of tracks) if (t.active) { t.x -= mx; t.y -= my; t.matched = false; t.age++; }

    if (cands) {
      const K = Math.min(this.baseline, (this._count - 1) >> 1);
      const B = this._ring[(head - 2 * K + N) % N];
      const used = new Uint8Array(cands.length);
      // Greedy nearest-neighbour association, closest pairs first.
      for (;;) {
        let bd = Infinity, bt = null, bi = -1;
        for (const t of tracks) {
          if (!t.active || t.matched) continue;
          for (let i = 0; i < cands.length; i++) {
            if (used[i]) continue;
            const c = cands[i];
            const d = Math.hypot((c.x + c.x1) / 2 - t.x, (c.y + c.y1) / 2 - t.y);
            const gate = this.matchDist + 0.5 * Math.max(t.w, t.h, c.w, c.h);
            if (d < gate && d < bd) { bd = d; bt = t; bi = i; }
          }
        }
        if (!bt) break;
        const c = cands[bi];
        used[bi] = 1; bt.matched = true;
        bt.x = (c.x + c.x1) / 2; bt.y = (c.y + c.y1) / 2;
        bt.w = c.w; bt.h = c.h; bt.area = c.area; bt.contrast = c.contrast; bt.pol = c.pol < 0 ? -1 : 1;
        bt.hits++; bt.missed = 0; bt.held = 0;
        const d = Math.hypot(bt.x + rx - bt.qx0, bt.y + ry - bt.qy0);
        if (d > bt.travel) bt.travel = d;
        this._snap(bt, cur, B, this._offB, this._lutB);
      }
      for (let i = 0; i < cands.length; i++) {
        if (used[i]) continue;
        const c = cands[i];
        let slot = null;
        for (const t of tracks) if (!t.active) { slot = t; break; }
        if (!slot) break;
        slot.active = true; slot.id = this._nextId++;
        slot.x = (c.x + c.x1) / 2; slot.y = (c.y + c.y1) / 2; slot.w = c.w; slot.h = c.h;
        slot.area = c.area; slot.contrast = c.contrast; slot.pol = c.pol < 0 ? -1 : 1;
        slot.hits = 1; slot.age = 1; slot.missed = 0; slot.held = 0; slot.confirmed = false; slot.matched = true;
        slot.qx0 = slot.x + rx; slot.qy0 = slot.y + ry; slot.travel = 0;
        this._snap(slot, cur, B, this._offB, this._lutB);
      }
    }

    const out = [];
    const hw = this._hw, hh = this._hh;
    for (const t of tracks) {
      if (!t.active) continue;
      if (!t.matched) {
        // No residual: resting (appearance still there) or gone.
        const holdBudget = Math.min(this.holdMax, Math.max(this.holdFloor, this.holdPerHit * t.hits));
        if (t.confirmed && t.held < holdBudget && this._stillThere(t, cur)) { t.held++; t.missed = 0; }
        else if (++t.missed > this.maxMissed) { t.active = false; continue; }
      }
      if (t.x < 0 || t.y < 0 || t.x >= hw || t.y >= hh) { t.active = false; continue; }
      if (!t.confirmed && t.hits >= this.minHits && t.travel >= this.minTravel) t.confirmed = true;
      if (!t.confirmed || t.missed > 1) continue;
      const bw = Math.max(t.w, 4), bh = Math.max(t.h, 4);
      out.push({
        x: Math.round((t.x - bw / 2) * 2), y: Math.round((t.y - bh / 2) * 2), w: Math.round(bw * 2), h: Math.round(bh * 2),
        area: t.area * 4, contrast: t.contrast, label: '', confidence: -1, source: 'motion',
        slow: true, held: t.held, slowId: t.id,
      });
    }
    if (dbg) dbg.info.tracks = tracks.filter(t => t.active).map(t => ({ id: t.id, x: t.x * 2, y: t.y * 2, hits: t.hits, age: t.age, conf: t.confirmed, held: t.held, missed: t.missed, travel: t.travel * 2, sig: t.sig, maskN: t.maskN }));
    return out;
  }
}
