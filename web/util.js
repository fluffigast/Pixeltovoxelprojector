// util.js — small shared helpers

/** OffscreenCanvas where available (Safari < 16.4 lacks it), else a detached <canvas>. */
export function createCanvas(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  return c;
}
