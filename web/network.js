// network.js — WebSocket client for multi-phone detection network

const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 15000;

export class Network {
  constructor() {
    this.ws = null;
    this.role = null;
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

    this._reconnectTimer = null;
    this._reconnectDelay = RECONNECT_MIN_MS;
    this._url = null;
  }

  /** @throws {Error} if the URL is not a valid ws:// or wss:// address */
  connect(url, role, deviceName) {
    let parsed;
    try { parsed = new URL(url); } catch (err) { throw new Error('Not a valid URL: ' + url, { cause: err }); }
    if (parsed.protocol !== 'ws:' && parsed.protocol !== 'wss:') {
      throw new Error('Server URL must start with ws:// or wss://');
    }

    this.disconnect();
    this._url = parsed.href;
    this.role = role;
    this.deviceName = deviceName || `Device ${Math.random().toString(36).slice(2, 6)}`;
    this._reconnectDelay = RECONNECT_MIN_MS;
    this._setState('connecting');
    this._open();
  }

  _setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.connected = state === 'connected';
    if (this.onStateChange) this.onStateChange(state);
  }

  _open() {
    let ws;
    try {
      ws = new WebSocket(this._url);
    } catch {
      // e.g. ws:// to a remote host from an https:// page (mixed content)
      this._scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this._reconnectDelay = RECONNECT_MIN_MS;
      ws.send(JSON.stringify({ type: 'register', role: this.role, name: this.deviceName }));
      this._setState('connected');
    };

    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      switch (msg.type) {
        case 'sources':
          if (this.onSources) this.onSources(msg.sources || []);
          break;
        case 'frame':
          if (this.onFrame) this.onFrame(msg.sourceId, msg.sourceName, msg.jpeg, msg.detections, msg.width, msg.height);
          break;
        case 'detections':
          if (this.onDetections) this.onDetections(msg.sourceId, msg.sourceName, msg.detections || [], msg.width, msg.height);
          break;
      }
    };

    // An error is always followed by close, so reconnecting is handled there.
    ws.onclose = () => {
      if (this.ws !== ws) return;   // superseded by disconnect() or a newer socket
      this.ws = null;
      this._scheduleReconnect();
    };
  }

  _scheduleReconnect() {
    this._setState('reconnecting');
    if (this._reconnectTimer) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._open();
    }, this._reconnectDelay);
    this._reconnectDelay = Math.min(this._reconnectDelay * 2, RECONNECT_MAX_MS);
  }

  /** Bytes queued but not yet sent; callers skip frames while this is large. */
  get bufferedAmount() {
    return this.ws ? this.ws.bufferedAmount : 0;
  }

  _send(msg) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  sendFrame(jpegBase64, detections, width, height) {
    return this._send({ type: 'frame', jpeg: jpegBase64, detections, width, height });
  }

  sendDetections(detections, width, height) {
    return this._send({ type: 'detections', detections, width, height });
  }

  /** Close the connection and stop reconnecting. */
  disconnect() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    const ws = this.ws;
    this.ws = null;
    if (ws) ws.close();
    this._setState('disconnected');
  }
}
