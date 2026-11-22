// server.js — Static file server + WebSocket relay for multi-phone detection
//
// Usage: node server.js [port]        (or PORT=8080 node server.js)
// Default port: 8080
//
// Protocol:
//   Client sends JSON messages:
//     { type: "register", role: "source"|"viewer", name: "My Phone" }
//     { type: "frame", jpeg: "<base64>", detections: [...], width, height }
//     { type: "detections", detections: [...], width, height }
//
//   Server broadcasts to viewers:
//     { type: "sources", sources: [{ id, name, connected }] }
//     { type: "frame", sourceId, sourceName, jpeg, detections, width, height }
//     { type: "detections", sourceId, sourceName, detections, width, height }

import { WebSocketServer } from 'ws';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = parseInt(process.argv[2] ?? process.env.PORT, 10) || 8080;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.mp4':  'video/mp4',
  '.webm': 'video/webm',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.svg':  'image/svg+xml',
  '.wasm': 'application/wasm',
};

const WEB_DIR = fileURLToPath(new URL('.', import.meta.url));
const WEB_ROOT = WEB_DIR.endsWith(sep) ? WEB_DIR : WEB_DIR + sep;

const MAX_PAYLOAD = 4 * 1024 * 1024;     // largest accepted WebSocket message
const MAX_NAME_LENGTH = 64;
const MAX_DETECTIONS = 500;
const MAX_BUFFERED = 8 * 1024 * 1024;    // skip viewers that cannot keep up
const HEARTBEAT_MS = 30000;
const ROLES = new Set(['source', 'viewer']);

function send(res, status, text) {
  res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8' });
  res.end(text);
}

/** Map a request URL to a file inside WEB_DIR, or null if it points anywhere else. */
function resolveRequestPath(url) {
  let pathname;
  try {
    pathname = decodeURIComponent(url.split('?')[0]);
  } catch {
    return null;
  }
  if (pathname.includes('\0')) return null;
  if (pathname.endsWith('/')) pathname += 'index.html';

  // resolve() collapses "..", so anything that ends up outside the web root is
  // refused, as are dotfiles (.env, .git) and node_modules.
  const filePath = resolve(WEB_ROOT, '.' + pathname);
  if (!filePath.startsWith(WEB_ROOT)) return null;
  const parts = filePath.slice(WEB_ROOT.length).split(sep);
  if (parts.some(part => part.startsWith('.')) || parts[0] === 'node_modules') return null;
  return filePath;
}

const httpServer = createServer(async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.setHeader('Allow', 'GET, HEAD');
    return send(res, 405, 'Method not allowed');
  }

  const filePath = resolveRequestPath(req.url);
  if (!filePath) return send(res, 404, 'Not found');

  let info;
  try {
    info = await stat(filePath);
  } catch {
    return send(res, 404, 'Not found');
  }
  if (!info.isFile()) return send(res, 404, 'Not found');

  const headers = {
    'Content-Type': MIME[extname(filePath).toLowerCase()] || 'application/octet-stream',
    'Accept-Ranges': 'bytes',
    'X-Content-Type-Options': 'nosniff',
  };

  // Byte ranges: required by Safari for <video>, and what makes seeking work.
  let start = 0;
  let end = info.size - 1;
  let status = 200;
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
  if (range && (range[1] || range[2])) {
    if (range[1]) {
      start = Number(range[1]);
      if (range[2]) end = Math.min(Number(range[2]), info.size - 1);
    } else {
      start = Math.max(0, info.size - Number(range[2]));   // suffix: last N bytes
    }
    if (start > end || start >= info.size) {
      res.writeHead(416, { 'Content-Range': `bytes */${info.size}` });
      return res.end();
    }
    status = 206;
    headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
  }
  headers['Content-Length'] = info.size === 0 ? 0 : end - start + 1;

  res.writeHead(status, headers);
  if (req.method === 'HEAD' || info.size === 0) return res.end();

  const stream = createReadStream(filePath, { start, end });
  stream.on('error', () => res.destroy());
  res.on('close', () => stream.destroy());
  stream.pipe(res);
});

const wss = new WebSocketServer({ server: httpServer, maxPayload: MAX_PAYLOAD });

let nextId = 1;
const clients = new Map(); // ws -> { id, role, name, alive }

function sendToViewers(payload) {
  const msg = JSON.stringify(payload);
  for (const [ws, info] of clients) {
    if (info.role !== 'viewer' || ws.readyState !== ws.OPEN) continue;
    if (ws.bufferedAmount > MAX_BUFFERED) continue;   // drop rather than queue without bound
    ws.send(msg);
  }
}

function broadcastSourceList() {
  const sources = [];
  for (const info of clients.values()) {
    if (info.role === 'source') sources.push({ id: info.id, name: info.name, connected: true });
  }
  sendToViewers({ type: 'sources', sources });
}

const toDimension = (v) => (Number.isFinite(v) && v > 0 ? Math.round(v) : undefined);
const toDetections = (v) => (Array.isArray(v) ? v.slice(0, MAX_DETECTIONS) : []);

wss.on('connection', (ws) => {
  const info = { id: nextId++, role: null, name: '', alive: true };
  info.name = `Device ${info.id}`;
  clients.set(ws, info);
  console.log(`Client ${info.id} connected`);

  ws.on('pong', () => { info.alive = true; });
  ws.on('error', (err) => console.warn(`Client ${info.id} error: ${err.message}`));

  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (!msg || typeof msg !== 'object') return;

    switch (msg.type) {
      case 'register': {
        info.role = ROLES.has(msg.role) ? msg.role : 'source';
        if (typeof msg.name === 'string' && msg.name.trim()) {
          info.name = msg.name.trim().slice(0, MAX_NAME_LENGTH);
        }
        console.log(`Client ${info.id} registered as ${info.role}: ${info.name}`);
        broadcastSourceList();
        break;
      }

      case 'frame': {
        // Relay frame from source to all viewers
        if (info.role !== 'source' || typeof msg.jpeg !== 'string') return;
        sendToViewers({
          type: 'frame',
          sourceId: info.id,
          sourceName: info.name,
          jpeg: msg.jpeg,
          detections: toDetections(msg.detections),
          width: toDimension(msg.width),
          height: toDimension(msg.height),
        });
        break;
      }

      case 'detections': {
        // Relay detections only (no frame)
        if (info.role !== 'source') return;
        sendToViewers({
          type: 'detections',
          sourceId: info.id,
          sourceName: info.name,
          detections: toDetections(msg.detections),
          width: toDimension(msg.width),
          height: toDimension(msg.height),
        });
        break;
      }
    }
  });

  ws.on('close', () => {
    console.log(`Client ${info.id} disconnected`);
    clients.delete(ws);
    if (info.role === 'source') broadcastSourceList();
  });
});

// Drop connections that vanished without a close frame (phone asleep, Wi-Fi lost).
const heartbeat = setInterval(() => {
  for (const [ws, info] of clients) {
    if (!info.alive) { ws.terminate(); continue; }
    info.alive = false;
    ws.ping();
  }
}, HEARTBEAT_MS);
wss.on('close', () => clearInterval(heartbeat));

httpServer.listen(PORT, () => {
  console.log(`BugSpotter server running on http://localhost:${PORT}`);
  console.log(`Open this URL on phones to connect cameras`);
  console.log(`Open with ?role=viewer for the dashboard`);
});
