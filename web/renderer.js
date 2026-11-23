// renderer.js — Polished canvas overlay for detection results

const CLASS_COLORS = {
  insect:  { stroke: '#FFD740', fill: 'rgba(255,215,64,' },
  drone:   { stroke: '#FF5252', fill: 'rgba(255,82,82,' },
  bird:    { stroke: '#40C4FF', fill: 'rgba(64,196,255,' },
  unknown: { stroke: '#B0B0B0', fill: 'rgba(176,176,176,' },   // moving, but not classified
};

// CSS px. Smallest ring drawn around a detection: about a fingertip on a phone.
const MIN_RING_RADIUS = 12;

// Aim ring for insects. The ring shows how well the aim is on the target, frame
// by frame, and adds up the time it was on:
//   ON   aim point inside the target's box: tight, solid, bright ring.
//        Only this time counts.
//   OFF  aim beside the target, or the target was not found this frame: the
//        ring is dashed and grows with the size of the miss. Nothing counts.
// "Aim point" is where the tracker predicted the target to be for this frame;
// the miss is its distance from where the target was actually found (see
// tracker.js). The label shows time on target / time needed. At 100% a flash
// marks it and the ring stays solid.
// The tracker often loses a target for a moment and finds it again under a new
// id (crawling insects, crowds). A new id that shows up where a target has just
// vanished takes over that target's time, so it is not lost.
// This is measurement and drawing only: nothing here switches a laser, and it
// cannot tell whether a real shot landed.
const AIM_OFF_MIN = 6;           // CSS px the ring is wider than ON as soon as the aim is off
const AIM_GAIN = 2;              // extra CSS px of radius per px of miss
const AIM_OFF_MAX = 32;          // most the ring grows; also used when the miss is unknown
const RING_EASE_MS = 40;         // how quickly the ring size follows a change in aim
const FLASH_MS = 220;            // length of the flash at 100%
const LOCK_FORGET_MS = 500;      // a target gone for longer than this starts over at 0%
const HANDOVER_RADIUS = 30;      // CSS px within which a new id continues a vanished target
const MAX_STEP_MS = 250;         // most time credited between two draws

let dwellNeeded = 500;

/** Time on target (ms, 50–60000) that counts as 100%. Applies to every overlay. */
export function setDwellNeeded(ms) {
  dwellNeeded = Math.min(60000, Math.max(50, Math.round(Number(ms)) || 500));
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
    this._locks = new Map();   // detection id -> aim state, see _aim()
  }

  /**
   * Aim state of a detection: { onMs, r, doneAt } carried from draw to draw
   * (null if it has no id). Adds the time since the last draw to `onMs` when
   * the aim is on the target.
   * @param {Set} present ids drawn in this call; their state is not up for takeover
   */
  _aim(det, cx, cy, now, present) {
    if (det.id === undefined || det.id === null) return null;
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
        lock = { onMs: 0, seen: now, x: cx, y: cy, r: null, doneAt: null };
      }
      this._locks.set(det.id, lock);
    }
    const dt = Math.min(now - lock.seen, MAX_STEP_MS);
    lock.dt = dt;
    if (det.onTarget && !det.missed && lock.doneAt === null) {
      lock.onMs += dt;
      if (lock.onMs >= dwellNeeded) lock.doneAt = now;
    }
    lock.seen = now;
    lock.x = cx;
    lock.y = cy;
    return lock;
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
   * @param detections boxes in overlay CSS pixels. For insects, an `id` that stays
   *   the same from call to call plus `onTarget` and `aimError` (CSS px) drive the
   *   aim ring and its time-on-target count
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
        // Only insects are aimed at; other small marks keep a plain ring.
        const aim = det.label === 'insect' ? this._aim(det, cx, cy, now, present) : null;
        const done = !!aim && aim.doneAt !== null;
        const on = !aim || done || (det.onTarget && !det.missed);

        // Ring size: tight when on target, wider the further the aim is off.
        let r = steady;
        if (aim) {
          let target = steady;
          if (!on) {
            const miss = Number.isFinite(det.aimError) && !det.missed ? AIM_OFF_MIN + AIM_GAIN * det.aimError : AIM_OFF_MAX;
            target = steady + Math.min(AIM_OFF_MAX, miss);
          }
          aim.r = aim.r === null ? target : aim.r + (target - aim.r) * (1 - Math.exp(-aim.dt / RING_EASE_MS));
          r = aim.r;
        }

        ctx.globalAlpha = 0.55 * fadeAlpha;
        ctx.strokeStyle = '#000';
        ctx.lineWidth = aim && on ? 5 : 4;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Solid and bright = on target; dashed = off. The dash pattern carries
        // the state on its own, without relying on colour.
        ctx.globalAlpha = (on ? 1 : 0.8) * fadeAlpha;
        ctx.strokeStyle = aim && on ? '#fff' : colors.stroke;
        ctx.lineWidth = aim && on ? 3 : 2;
        if (!on) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.globalAlpha = 0.95 * fadeAlpha;
        ctx.fillStyle = colors.stroke;
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, Math.PI * 2);
        ctx.fill();

        if (aim) {
          // Reaching 100%: a bright core and a burst ring that expands and fades.
          const sinceDone = done ? now - aim.doneAt : Infinity;
          if (sinceDone < FLASH_MS) {
            const p = sinceDone / FLASH_MS;
            ctx.globalAlpha = (1 - p) * fadeAlpha;
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(cx, cy, 2 + 5 * (1 - p), 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 3 * (1 - p) + 1;
            ctx.beginPath();
            ctx.arc(cx, cy, steady + 14 * p, 0, Math.PI * 2);
            ctx.stroke();
          }

          // Time on target as a share of the time needed.
          const text = (done ? 100 : Math.min(99, Math.floor(100 * aim.onMs / dwellNeeded))) + '%';
          ctx.font = '600 11px system-ui, sans-serif';
          ctx.globalAlpha = 0.9 * fadeAlpha;
          ctx.lineWidth = 3;
          ctx.strokeStyle = '#000';
          ctx.strokeText(text, cx + steady + 4, cy - steady + 3);
          ctx.fillStyle = done ? '#fff' : colors.stroke;
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
