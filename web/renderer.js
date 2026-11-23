// renderer.js — Polished canvas overlay for detection results

const CLASS_COLORS = {
  insect:  { stroke: '#FFD740', fill: 'rgba(255,215,64,' },
  drone:   { stroke: '#FF5252', fill: 'rgba(255,82,82,' },
  bird:    { stroke: '#40C4FF', fill: 'rgba(64,196,255,' },
  unknown: { stroke: '#B0B0B0', fill: 'rgba(176,176,176,' },   // moving, but not classified
};

// CSS px. Smallest ring drawn around a detection: about a fingertip on a phone.
const MIN_RING_RADIUS = 12;

// Lock-on animation for the ring. For each target it repeats:
//   the ring starts wide and closes in on the target (LOCK_MS), then a short
//   flash marks a shot (SHOT_MS). Every shot counts as one hit.
// A label shows hits / hits needed as a percentage. At 100% the cycle stops and
// the normal steady ring stays on the target for as long as it is tracked.
// The tracker often loses a target for a moment and finds it again under a new
// id (crawling insects, crowds). A new id that shows up where a target has just
// vanished takes over that target's progress, so its hits are not lost.
// This is drawing only: nothing here knows whether a real shot was fired or landed.
const LOCK_MS = 800;             // time the ring takes to close in
const LOCK_START_RADIUS = 44;    // CSS px, ring size at the start of each cycle
const LOCK_END_RADIUS = 3;       // CSS px, ring size at the moment of the shot
const SHOT_MS = 220;             // length of the flash
const CYCLE_MS = LOCK_MS + SHOT_MS;
const LOCK_FORGET_MS = 500;      // a target gone for longer than this starts over at 0%
const HANDOVER_RADIUS = 30;      // CSS px within which a new id continues a vanished target
const MAX_STEP_MS = 250;         // most time credited between two draws

let hitsNeeded = 3;

/** How many hits on one target count as 100% (1–50). Applies to every overlay. */
export function setHitsNeeded(n) {
  hitsNeeded = Math.min(50, Math.max(1, Math.round(Number(n)) || 1));
}

const STAT_IDS = [
  'fpsDisplay', 'countDisplay', 'insectCount', 'droneCount', 'birdCount',
  'inferDisplay', 'backendDisplay', 'motionMsDisplay', 'resolutionDisplay',
];

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = 0;    // CSS pixels
    this.height = 0;
    this._statEls = null;   // looked up on first use; overlay-only renderers never need them
    this._statText = {};
    this._locks = new Map();   // detection id -> { active, seen (ms), x, y }, drives the lock-on animation
  }

  /**
   * Milliseconds this detection has been on screen so far (Infinity if it has no id).
   * @param {Set} present ids drawn in this call; their progress is not up for takeover
   */
  _lockAge(det, cx, cy, now, present) {
    if (det.id === undefined || det.id === null) return Infinity;   // no identity: steady ring
    let lock = this._locks.get(det.id);
    if (lock && now - lock.seen > LOCK_FORGET_MS) lock = null;       // back after being gone: start over
    if (!lock) {
      // New id. Continue the nearest target that has just vanished, if there is one.
      let from = null;
      let best = HANDOVER_RADIUS;
      for (const [id, other] of this._locks) {
        if (present.has(id) || now - other.seen > LOCK_FORGET_MS) continue;
        const dist = Math.hypot(other.x - cx, other.y - cy);
        if (dist <= best) { best = dist; from = id; }
      }
      if (from !== null) {
        lock = this._locks.get(from);
        this._locks.delete(from);
      } else {
        lock = { active: 0, seen: now, x: cx, y: cy };
      }
      this._locks.set(det.id, lock);
    }
    lock.active += Math.min(now - lock.seen, MAX_STEP_MS);
    lock.seen = now;
    lock.x = cx;
    lock.y = cy;
    return lock.active;
  }

  /** Size the overlay in CSS pixels; the backing store follows devicePixelRatio. */
  resize(width, height) {
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(width * dpr);
    const bh = Math.round(height * dpr);
    if (this.canvas.width !== bw || this.canvas.height !== bh) {
      this.canvas.width = bw;
      this.canvas.height = bh;
    }
    this.width = width;
    this.height = height;
    // Draw calls below use CSS pixels.
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /**
   * @param detections boxes in overlay CSS pixels; an `id` that stays the same
   *   from call to call lets the ring animate and count hits (see LOCK_MS)
   * @param {number} [now] timestamp in ms, same clock on every call
   */
  draw(detections, now = performance.now()) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const sorted = detections.toSorted((a, b) => (a.w * a.h) - (b.w * b.h));
    const present = new Set(detections.map(d => d.id));

    for (const det of sorted) {
      const colors = CLASS_COLORS[det.label] || CLASS_COLORS.unknown;
      const { x, y, w, h } = det;
      const cx = x + w / 2;
      const cy = y + h / 2;
      const area = w * h;
      const fadeAlpha = det.missed ? Math.max(0.1, 1 - det.missed * 0.05) : 1;

      // --- Flight path: makes a fast, tiny target readable at a glance ---
      if (det.trail && det.trail.length > 1) {
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 1.5;
        ctx.lineJoin = 'round';
        const n = det.trail.length;
        for (let i = 1; i < n; i++) {
          ctx.globalAlpha = 0.6 * fadeAlpha * (i / n);   // older segments fade out
          ctx.beginPath();
          ctx.moveTo(det.trail[i - 1][0], det.trail[i - 1][1]);
          ctx.lineTo(det.trail[i][0], det.trail[i][1]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // --- Small detections: a ring big enough to find on a phone screen ---
      // A fly is often only a few screen pixels across; the mark must not be.
      // Dark halo under the coloured ring so it reads on bright sky as well as
      // on dark foliage.
      if (area < 1600) {
        const steady = Math.max(MIN_RING_RADIUS, Math.max(w, h) / 2 + 5);
        const age = this._lockAge(det, cx, cy, now, present);
        const tracked = age !== Infinity;

        // Where in the closing-in / shot cycle this target is, and its hits so far.
        const done = age >= hitsNeeded * CYCLE_MS;
        const cycle = done ? hitsNeeded : Math.floor(age / CYCLE_MS);
        const sinceShot = done ? Infinity : age - cycle * CYCLE_MS - LOCK_MS;
        const flashing = sinceShot >= 0 && sinceShot < SHOT_MS;
        const hits = done ? hitsNeeded : cycle + (sinceShot >= 0 ? 1 : 0);

        // Closing in: the ring shrinks from wide to a point on the target. It is
        // hidden during a flash, except after the last hit, when the steady ring
        // fades in and stays.
        let r = steady;
        let ringAlpha = 1;
        if (sinceShot < 0) {
          r = LOCK_START_RADIUS + (LOCK_END_RADIUS - LOCK_START_RADIUS) * (1 + sinceShot / LOCK_MS);
        } else if (flashing) {
          ringAlpha = hits >= hitsNeeded ? sinceShot / SHOT_MS : 0;
        }

        if (ringAlpha > 0) {
          ctx.globalAlpha = 0.55 * fadeAlpha * ringAlpha;
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.stroke();

          ctx.globalAlpha = 0.95 * fadeAlpha * ringAlpha;
          ctx.strokeStyle = colors.stroke;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.globalAlpha = 0.95 * fadeAlpha;
        ctx.fillStyle = colors.stroke;
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, Math.PI * 2);
        ctx.fill();

        // The shot: a bright core and a burst ring that expands and fades.
        if (flashing) {
          const p = sinceShot / SHOT_MS;
          ctx.globalAlpha = (1 - p) * fadeAlpha;
          ctx.fillStyle = '#fff';
          ctx.beginPath();
          ctx.arc(cx, cy, 2 + 5 * (1 - p), 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 3 * (1 - p) + 1;
          ctx.beginPath();
          ctx.arc(cx, cy, LOCK_END_RADIUS + (steady + 10 - LOCK_END_RADIUS) * p, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Hits so far as a share of the hits needed.
        if (tracked) {
          const text = Math.round(100 * hits / hitsNeeded) + '%';
          ctx.font = '600 11px system-ui, sans-serif';
          ctx.globalAlpha = 0.9 * fadeAlpha;
          ctx.lineWidth = 3;
          ctx.strokeStyle = '#000';
          ctx.strokeText(text, cx + steady + 4, cy - steady + 3);
          ctx.fillStyle = hits >= hitsNeeded ? '#fff' : colors.stroke;
          ctx.fillText(text, cx + steady + 4, cy - steady + 3);
        }
        ctx.globalAlpha = 1;
        continue;
      }

      // --- Medium+ detections: bracket corners + label ---
      ctx.globalAlpha = 0.85 * fadeAlpha;
      ctx.strokeStyle = colors.stroke;

      // Subtle fill
      ctx.fillStyle = colors.fill + (0.08 * fadeAlpha) + ')';
      ctx.fillRect(x, y, w, h);

      // Corner brackets (top-left, top-right, bottom-left, bottom-right)
      const cl = Math.min(12, w / 3, h / 3);
      ctx.lineWidth = 2;
      ctx.beginPath();
      // TL
      ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y);
      // TR
      ctx.moveTo(x + w - cl, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cl);
      // BL
      ctx.moveTo(x, y + h - cl); ctx.lineTo(x, y + h); ctx.lineTo(x + cl, y + h);
      // BR
      ctx.moveTo(x + w - cl, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cl);
      ctx.stroke();

      // Thin connecting lines between corners
      ctx.lineWidth = 0.5;
      ctx.globalAlpha = 0.3 * fadeAlpha;
      ctx.strokeRect(x, y, w, h);

      // Label
      ctx.globalAlpha = 0.85 * fadeAlpha;
      if (det.confidence > 0 || area > 3000) {
        const label = det.confidence > 0
          ? det.label + ' ' + Math.round(det.confidence * 100) + '%'
          : det.label || '';
        if (label) {
          ctx.font = '600 11px system-ui, sans-serif';
          const tm = ctx.measureText(label);
          const pad = 5;
          const lh = 16;
          const lx = x;
          const ly = y > lh + 4 ? y - lh - 3 : y + h + 3;

          // Label background with rounded corners
          const lw = tm.width + pad * 2;
          ctx.globalAlpha = 0.75 * fadeAlpha;
          ctx.fillStyle = colors.stroke;
          ctx.beginPath();
          ctx.roundRect(lx, ly, lw, lh, 3);
          ctx.fill();

          ctx.globalAlpha = 1 * fadeAlpha;
          ctx.fillStyle = '#000';
          ctx.fillText(label, lx + pad, ly + 11.5);
        }
      }
      ctx.globalAlpha = 1;
    }

    // Forget targets that are gone, so one that comes back is locked onto again.
    for (const [id, lock] of this._locks) {
      if (now - lock.seen > LOCK_FORGET_MS) this._locks.delete(id);
    }
  }

  /** Update the sidebar. Only touches the DOM for values that changed. */
  updateStats(stats) {
    if (!this._statEls) {
      this._statEls = {};
      for (const id of STAT_IDS) this._statEls[id] = document.getElementById(id);
    }
    const set = (id, text) => {
      text = String(text);
      if (this._statText[id] === text) return;
      this._statText[id] = text;
      const el = this._statEls[id];
      if (el) el.textContent = text;
    };
    set('fpsDisplay', stats.fps ?? '--');
    set('countDisplay', stats.count);
    set('backendDisplay', stats.backend);
    set('inferDisplay', stats.inferMs ? stats.inferMs + 'ms' : '--');
    set('motionMsDisplay', stats.motionMs != null ? stats.motionMs.toFixed(1) + 'ms' : '--');
    set('resolutionDisplay', stats.resolution || '--');
    set('insectCount', stats.classCounts?.insect ?? 0);
    set('droneCount', stats.classCounts?.drone ?? 0);
    set('birdCount', stats.classCounts?.bird ?? 0);

    const fpsTile = this._statEls.fpsDisplay?.parentElement;
    if (fpsTile) fpsTile.classList.toggle('stat-warn', stats.fps != null && stats.fps < 5);
  }

  clear() {
    this.ctx.clearRect(0, 0, this.width, this.height);
    this._locks.clear();
  }
}
