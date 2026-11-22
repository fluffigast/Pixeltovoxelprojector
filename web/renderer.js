// renderer.js — Polished canvas overlay for detection results

const CLASS_COLORS = {
  insect:  { stroke: '#FFD740', fill: 'rgba(255,215,64,', glow: '#FFD740' },
  drone:   { stroke: '#FF5252', fill: 'rgba(255,82,82,',  glow: '#FF5252' },
  bird:    { stroke: '#40C4FF', fill: 'rgba(64,196,255,',  glow: '#40C4FF' },
  unknown: { stroke: '#FFD740', fill: 'rgba(255,215,64,', glow: '#FFD740' },
};

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
    this._statEls = {};
    this._statText = {};
    for (const id of STAT_IDS) this._statEls[id] = document.getElementById(id);
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

  /** @param detections boxes in overlay CSS pixels */
  draw(detections) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const sorted = detections.toSorted((a, b) => (a.w * a.h) - (b.w * b.h));

    for (const det of sorted) {
      const colors = CLASS_COLORS[det.label] || CLASS_COLORS.unknown;
      const { x, y, w, h } = det;
      const cx = x + w / 2;
      const cy = y + h / 2;
      const area = w * h;
      const fadeAlpha = det.missed ? Math.max(0.1, 1 - det.missed * 0.05) : 1;

      // --- Tiny detections: soft glow dot ---
      if (area < 225) {
        // Outer glow
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 10);
        grad.addColorStop(0, colors.fill + (0.5 * fadeAlpha) + ')');
        grad.addColorStop(1, colors.fill + '0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, 10, 0, Math.PI * 2);
        ctx.fill();

        // Core dot
        ctx.globalAlpha = 0.9 * fadeAlpha;
        ctx.fillStyle = colors.stroke;
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        continue;
      }

      // --- Small detections: ring with glow ---
      if (area < 1600) {
        const r = Math.max(w, h) / 2 + 5;

        // Glow behind ring
        const grad = ctx.createRadialGradient(cx, cy, r - 3, cx, cy, r + 6);
        grad.addColorStop(0, colors.fill + (0.15 * fadeAlpha) + ')');
        grad.addColorStop(1, colors.fill + '0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, r + 6, 0, Math.PI * 2);
        ctx.fill();

        // Ring
        ctx.globalAlpha = 0.8 * fadeAlpha;
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Center dot
        ctx.fillStyle = colors.stroke;
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, Math.PI * 2);
        ctx.fill();
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
  }

  /** Update the sidebar. Only touches the DOM for values that changed. */
  updateStats(stats) {
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
  }
}
