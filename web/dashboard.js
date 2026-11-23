// dashboard.js — Grid of live feeds from the devices connected to this one

import { Renderer } from './renderer.js';

const STALE_MS = 3000;                 // a feed with no frame for this long is dimmed
const MAX_JPEG_CHARS = 400000;         // ignore absurdly large frames
const JPEG_PREFIX = 'data:image/jpeg;base64,';

const finite = (v) => (Number.isFinite(v) ? v : 0);

/** Keep only well-formed boxes; remote data is not trusted. */
function cleanDetections(detections) {
  const out = [];
  for (const d of Array.isArray(detections) ? detections.slice(0, 500) : []) {
    if (!d || typeof d !== 'object') continue;
    out.push({
      id: Number.isFinite(d.id) ? d.id : undefined,
      x: finite(d.x), y: finite(d.y), w: finite(d.w), h: finite(d.h),
      label: typeof d.label === 'string' ? d.label.slice(0, 20) : 'unknown',
      confidence: finite(d.confidence),
      missed: finite(d.missed),
      trail: Array.isArray(d.trail)
        ? d.trail.slice(-16).filter(p => Array.isArray(p)).map(p => [finite(p[0]), finite(p[1])]) : null,
    });
  }
  return out;
}

export function countByClass(detections) {
  const counts = { insect: 0, drone: 0, bird: 0, unknown: 0 };
  for (const d of detections) {
    if (counts[d.label] !== undefined) counts[d.label]++;
    else counts.unknown++;
  }
  return counts;
}

export class Dashboard {
  /** @param {HTMLElement} container element the feed cards are placed in */
  constructor(container) {
    this.container = container;
    this.feeds = new Map();   // sourceId -> feed
    this._empty = document.createElement('p');
    this._empty.className = 'dashboard-empty';
    container.appendChild(this._empty);
    this.setEmptyMessage('Waiting for devices…');
    this._staleTimer = setInterval(() => this._markStale(), 1000);
  }

  setEmptyMessage(text) {
    this._empty.textContent = text;
  }

  /** Replace the list of connected devices: [{ id, name }]. */
  setSources(sources) {
    const ids = new Set(sources.map(s => s.id));
    for (const [id, feed] of this.feeds) {
      if (!ids.has(id)) {
        feed.root.remove();
        this.feeds.delete(id);
      }
    }
    for (const s of sources) this._feed(s.id, s.name);
    this._empty.hidden = this.feeds.size > 0;
  }

  _feed(id, name) {
    let feed = this.feeds.get(id);
    if (!feed) {
      const root = document.createElement('figure');
      root.className = 'feed stale';
      const picture = document.createElement('div');
      picture.className = 'feed-picture';
      const img = document.createElement('img');
      img.alt = '';
      const canvas = document.createElement('canvas');
      picture.append(img, canvas);
      const caption = document.createElement('figcaption');
      const title = document.createElement('span');
      title.className = 'feed-name';
      const summary = document.createElement('span');
      summary.className = 'feed-summary';
      summary.textContent = 'waiting for video…';
      caption.append(title, summary);
      root.append(picture, caption);
      this.container.appendChild(root);

      feed = { root, img, title, summary, renderer: new Renderer(canvas), detections: [], width: 0, height: 0, lastFrame: 0 };
      // The overlay is drawn once the picture it belongs to has been decoded.
      img.addEventListener('load', () => this._drawOverlay(feed));
      this.feeds.set(id, feed);
      this._empty.hidden = true;
    }
    if (name && feed.title.textContent !== name) feed.title.textContent = name;
    return feed;
  }

  /** Show a frame (JPEG data URL) and its detections for one device. */
  showFrame(id, name, jpeg, detections, width, height) {
    const feed = this._feed(id, name);
    feed.detections = cleanDetections(detections);
    feed.width = finite(width);
    feed.height = finite(height);
    feed.lastFrame = performance.now();
    feed.root.classList.remove('stale');
    if (typeof jpeg === 'string' && jpeg.startsWith(JPEG_PREFIX) && jpeg.length <= MAX_JPEG_CHARS) {
      feed.img.src = jpeg;
    } else {
      this._drawOverlay(feed);
    }
    const c = countByClass(feed.detections);
    feed.summary.textContent = c.insect + ' insect' + (c.insect === 1 ? '' : 's') +
      (c.drone ? ' · ' + c.drone + ' drone' + (c.drone === 1 ? '' : 's') : '') +
      (c.bird ? ' · ' + c.bird + ' bird' + (c.bird === 1 ? '' : 's') : '') +
      (c.unknown ? ' · ' + c.unknown + ' other' : '');
  }

  _drawOverlay(feed) {
    const w = feed.img.clientWidth;
    const h = feed.img.clientHeight;
    if (!w || !h || !feed.width || !feed.height) return;
    const sx = w / feed.width;
    const sy = h / feed.height;
    feed.renderer.resize(w, h);
    feed.renderer.draw(feed.detections.map(d => ({
      ...d,
      x: d.x * sx, y: d.y * sy, w: d.w * sx, h: d.h * sy,
      trail: d.trail && d.trail.map(p => [p[0] * sx, p[1] * sy]),
    })));
  }

  _markStale() {
    const now = performance.now();
    for (const feed of this.feeds.values()) {
      feed.root.classList.toggle('stale', now - feed.lastFrame > STALE_MS);
    }
  }

  /** Detections of every live feed, for combined statistics. */
  allDetections() {
    const now = performance.now();
    const all = [];
    for (const feed of this.feeds.values()) {
      if (now - feed.lastFrame <= STALE_MS) all.push(...feed.detections);
    }
    return all;
  }

  clear() {
    this.setSources([]);
  }
}
