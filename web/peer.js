// peer.js — Room-code links between devices, straight from the hosted page
//
// One device opens a dashboard and gets a room code; other devices join with
// that code and stream their frames and detections to it. Devices talk to each
// other directly over a WebRTC data channel. A public PeerJS broker is used
// only to introduce them, so no server of your own is needed and it works
// from an https page (which a phone camera requires).
//
// Same interface as Network (network.js), so the app can use either.

const PEERJS_URL = 'https://cdn.jsdelivr.net/npm/peerjs@1.5.5/dist/peerjs.min.js';
const ID_PREFIX = 'bugspotter-room-';
const ROOM_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';   // no 0/O, 1/I
export const ROOM_LENGTH = 6;
const RETRY_MIN_MS = 1500;
const RETRY_MAX_MS = 10000;

export function newRoomCode() {
  const bytes = crypto.getRandomValues(new Uint8Array(ROOM_LENGTH));
  return Array.from(bytes, b => ROOM_ALPHABET[b % ROOM_ALPHABET.length]).join('');
}

/** Upper-case and strip separators; returns '' if what is left is not a room code. */
export function normalizeRoomCode(text) {
  const code = String(text || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  return code.length === ROOM_LENGTH && [...code].every(c => ROOM_ALPHABET.includes(c)) ? code : '';
}

let peerLibrary = null;
function loadPeerLibrary() {
  if (!peerLibrary) {
    peerLibrary = new Promise((resolve, reject) => {
      if (window.Peer) return resolve(window.Peer);
      const script = document.createElement('script');
      script.src = PEERJS_URL;
      script.onload = () => (window.Peer ? resolve(window.Peer) : reject(new Error('PeerJS did not load')));
      script.onerror = () => reject(new Error('Could not load the connection library (offline?)'));
      document.head.appendChild(script);
    }).catch((e) => { peerLibrary = null; throw e; });
  }
  return peerLibrary;
}

export class PeerLink {
  /**
   * @param {object} [options]
   * @param {string} [options.broker] "host:port" of a self-hosted PeerServer,
   *   used instead of the public PeerJS broker
   */
  constructor(options = {}) {
    this.broker = options.broker || null;
    this.role = null;
    this.room = null;
    this.deviceName = null;
    this.connected = false;
    /** 'disconnected' | 'connecting' | 'connected' | 'reconnecting' */
    this.state = 'disconnected';
    /** Human-readable reason for the current state, when there is one. */
    this.detail = '';

    this.onStateChange = null; // callback: (state) => {}
    this.onSources = null;     // callback: (sources) => {}
    this.onFrame = null;       // callback: (sourceId, sourceName, jpeg, detections, width, height) => {}
    this.onDetections = null;  // callback: (sourceId, sourceName, detections, width, height) => {}

    this._peer = null;
    this._hub = null;               // source: the connection to the dashboard
    this._sources = new Map();      // dashboard: connection -> { id, name }
    this._nextSourceId = 1;
    this._retryTimer = null;
    this._retryDelay = RETRY_MIN_MS;
    this._session = 0;              // bumped by disconnect() so stale callbacks bail out
  }

  /**
   * @param {string} room room code
   * @param {'viewer'|'source'} role 'viewer' opens the room, 'source' joins it
   */
  connect(room, role, deviceName) {
    const code = normalizeRoomCode(room);
    if (!code) throw new Error('Room codes are ' + ROOM_LENGTH + ' letters or digits');
    this.disconnect();
    this.room = code;
    this.role = role;
    this.deviceName = deviceName || 'Device';
    this._retryDelay = RETRY_MIN_MS;
    this._setState('connecting');
    this._open();
  }

  _setState(state, detail = '') {
    const changed = this.state !== state || this.detail !== detail;
    this.state = state;
    this.detail = detail;
    this.connected = state === 'connected';
    if (changed && this.onStateChange) this.onStateChange(state);
  }

  _peerOptions() {
    if (!this.broker) return {};
    const [host, port] = this.broker.split(':');
    const secure = location.protocol === 'https:';
    return { host, port: Number(port) || (secure ? 443 : 80), path: '/', secure };
  }

  async _open() {
    const session = this._session;
    let Peer;
    try {
      Peer = await loadPeerLibrary();
    } catch (e) {
      if (session === this._session) this._retry(e.message);
      return;
    }
    if (session !== this._session) return;

    const hostId = ID_PREFIX + this.room;
    const peer = this.role === 'viewer' ? new Peer(hostId, this._peerOptions()) : new Peer(this._peerOptions());
    this._peer = peer;
    const current = () => session === this._session && this._peer === peer;

    peer.on('open', () => {
      if (!current()) return;
      if (this.role === 'viewer') {
        this._retryDelay = RETRY_MIN_MS;
        this._setState('connected');
        this._emitSources();
      } else {
        this._join(peer, hostId, current);
      }
    });

    // Dashboard: a device joined.
    peer.on('connection', (conn) => {
      if (!current() || this.role !== 'viewer') { conn.close(); return; }
      const info = { id: this._nextSourceId++, name: 'Device' };
      conn.on('open', () => { this._sources.set(conn, info); this._emitSources(); });
      conn.on('data', (msg) => this._onSourceMessage(info, msg));
      const gone = () => { if (this._sources.delete(conn)) this._emitSources(); };
      conn.on('close', gone);
      conn.on('error', gone);
    });

    peer.on('disconnected', () => {
      // Lost the broker. Existing device-to-device links keep working; ask to be
      // listed again so new devices can still find this one.
      if (current() && !peer.destroyed) peer.reconnect();
    });

    peer.on('error', (err) => {
      if (!current()) return;
      if (err.type === 'unavailable-id') {
        this._fail('Room ' + this.room + ' already has a dashboard. Close it or open a new room.');
      } else if (err.type === 'peer-unavailable') {
        this._retry('No dashboard open for room ' + this.room + ' yet');
      } else if (err.type === 'browser-incompatible') {
        this._fail('This browser cannot make device-to-device connections.');
      } else if (!this.connected) {
        this._retry('Connection service unreachable');
      }
    });
  }

  /** Source: connect to the dashboard that owns the room. */
  _join(peer, hostId, current) {
    // PeerJS's default binary serialisation splits large messages into chunks;
    // its JSON mode refuses anything over ~16 kB, which a video frame exceeds.
    const conn = peer.connect(hostId, { reliable: true });
    this._hub = conn;
    conn.on('open', () => {
      if (!current()) return;
      this._retryDelay = RETRY_MIN_MS;
      conn.send({ type: 'register', name: this.deviceName });
      this._setState('connected');
    });
    const lost = () => {
      if (!current() || this._hub !== conn) return;
      this._hub = null;
      this._retry('Dashboard went away');
    };
    conn.on('close', lost);
    conn.on('error', lost);
  }

  _onSourceMessage(info, msg) {
    if (!msg || typeof msg !== 'object') return;
    const detections = Array.isArray(msg.detections) ? msg.detections : [];
    switch (msg.type) {
      case 'register':
        if (typeof msg.name === 'string' && msg.name.trim()) info.name = msg.name.trim().slice(0, 64);
        this._emitSources();
        break;
      case 'frame':
        if (this.onFrame && typeof msg.jpeg === 'string') this.onFrame(info.id, info.name, msg.jpeg, detections, msg.width, msg.height);
        break;
      case 'detections':
        if (this.onDetections) this.onDetections(info.id, info.name, detections, msg.width, msg.height);
        break;
    }
  }

  _emitSources() {
    if (this.onSources) this.onSources([...this._sources.values()].map(s => ({ id: s.id, name: s.name, connected: true })));
  }

  /** Tear down and try again later (dashboard not open yet, network hiccup). */
  _retry(detail) {
    this._teardown();
    this._setState('reconnecting', detail);
    if (this._retryTimer) return;
    this._retryTimer = setTimeout(() => {
      this._retryTimer = null;
      this._open();
    }, this._retryDelay);
    this._retryDelay = Math.min(this._retryDelay * 1.5, RETRY_MAX_MS);
  }

  /** Give up: retrying cannot help (for example the room code is taken). */
  _fail(detail) {
    this.disconnect();
    this._setState('disconnected', detail);
  }

  _teardown() {
    const peer = this._peer;
    this._peer = null;
    this._hub = null;
    this._sources.clear();
    if (peer && !peer.destroyed) peer.destroy();
  }

  /** Bytes queued but not yet sent; callers skip frames while this is large. */
  get bufferedAmount() {
    return this._hub?.dataChannel?.bufferedAmount ?? 0;
  }

  _send(msg) {
    if (!this.connected || !this._hub?.open) return false;
    this._hub.send(msg);
    return true;
  }

  sendFrame(jpeg, detections, width, height) {
    return this._send({ type: 'frame', jpeg, detections, width, height });
  }

  sendDetections(detections, width, height) {
    return this._send({ type: 'detections', detections, width, height });
  }

  /** Leave the room and stop retrying. */
  disconnect() {
    this._session++;
    if (this._retryTimer) {
      clearTimeout(this._retryTimer);
      this._retryTimer = null;
    }
    this._teardown();
    this._setState('disconnected');
  }
}
