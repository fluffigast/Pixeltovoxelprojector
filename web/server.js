// server.js — WebSocket relay server for multi-phone detection
//
// Usage: node server.js [port]
// Default port: 8080
//
// Protocol:
//   Client sends JSON messages:
//     { type: "register", role: "source"|"viewer", name: "My Phone" }
//     { type: "frame", jpeg: "<base64>", detections: [...], width, height }
//     { type: "detections", detections: [...] }
//
//   Server broadcasts to viewers:
//     { type: "sources", sources: [{ id, name, connected }] }
//     { type: "frame", sourceId, sourceName, jpeg, detections, width, height }

import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';

const PORT = parseInt(process.argv[2]) || 8080;

// Serve static files for the web app
const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.mp4':  'video/mp4',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.svg':  'image/svg+xml',
  '.wasm': 'application/wasm',
};

const WEB_DIR = new URL('.', import.meta.url).pathname;

const httpServer = createServer((req, res) => {
  let urlPath = req.url.split('?')[0];
  if (urlPath === '/') urlPath = '/index.html';

  const filePath = join(WEB_DIR, urlPath);
  if (!existsSync(filePath)) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }

  const ext = extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';

  try {
    const data = readFileSync(filePath);
    res.writeHead(200, {
      'Content-Type': mime,
      'Access-Control-Allow-Origin': '*',
    });
    res.end(data);
  } catch {
    res.writeHead(500);
    res.end('Server error');
  }
});

const wss = new WebSocketServer({ server: httpServer });

let nextId = 1;
const clients = new Map(); // ws -> { id, role, name }

function broadcastSourceList() {
  const sources = [];
  for (const [, info] of clients) {
    if (info.role === 'source') {
      sources.push({ id: info.id, name: info.name, connected: true });
    }
  }
  const msg = JSON.stringify({ type: 'sources', sources });
  for (const [ws, info] of clients) {
    if (info.role === 'viewer' && ws.readyState === 1) {
      ws.send(msg);
    }
  }
}

wss.on('connection', (ws) => {
  const clientId = nextId++;
  clients.set(ws, { id: clientId, role: null, name: `Device ${clientId}` });

  console.log(`Client ${clientId} connected`);

  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }

    const info = clients.get(ws);
    if (!info) return;

    switch (msg.type) {
      case 'register':
        info.role = msg.role || 'source';
        info.name = msg.name || info.name;
        console.log(`Client ${info.id} registered as ${info.role}: ${info.name}`);
        broadcastSourceList();
        break;

      case 'frame':
        // Relay frame from source to all viewers
        if (info.role !== 'source') return;
        const frameMsg = JSON.stringify({
          type: 'frame',
          sourceId: info.id,
          sourceName: info.name,
          jpeg: msg.jpeg,
          detections: msg.detections || [],
          width: msg.width,
          height: msg.height,
        });
        for (const [vws, vinfo] of clients) {
          if (vinfo.role === 'viewer' && vws.readyState === 1) {
            vws.send(frameMsg);
          }
        }
        break;

      case 'detections':
        // Relay detections only (no frame)
        if (info.role !== 'source') return;
        const detMsg = JSON.stringify({
          type: 'detections',
          sourceId: info.id,
          sourceName: info.name,
          detections: msg.detections || [],
        });
        for (const [vws, vinfo] of clients) {
          if (vinfo.role === 'viewer' && vws.readyState === 1) {
            vws.send(detMsg);
          }
        }
        break;
    }
  });

  ws.on('close', () => {
    console.log(`Client ${clients.get(ws)?.id} disconnected`);
    clients.delete(ws);
    broadcastSourceList();
  });
});

httpServer.listen(PORT, () => {
  console.log(`BugSpotter server running on http://localhost:${PORT}`);
  console.log(`Open this URL on phones to connect cameras`);
  console.log(`Open with ?role=viewer for the dashboard`);
});
