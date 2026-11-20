// network.js — WebSocket client for multi-phone detection network

export class Network {
  constructor() {
    this.ws = null;
    this.role = null;
    this.deviceName = null;
    this.connected = false;
    this.onSources = null;   // callback: (sources) => {}
    this.onFrame = null;     // callback: (sourceId, sourceName, jpeg, detections, width, height) => {}
    this.onDetections = null; // callback: (sourceId, sourceName, detections) => {}
    this.onConnect = null;
    this.onDisconnect = null;
    this._reconnectTimer = null;
    this._url = null;
  }

  connect(url, role, deviceName) {
    this._url = url;
    this.role = role;
    this.deviceName = deviceName || `Device ${Math.random().toString(36).slice(2, 6)}`;

    this._doConnect();
  }

  _doConnect() {
    if (this.ws && this.ws.readyState <= 1) return;

    try {
      this.ws = new WebSocket(this._url);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.connected = true;
      this.ws.send(JSON.stringify({
        type: 'register',
        role: this.role,
        name: this.deviceName,
      }));
      if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
      }
      if (this.onConnect) this.onConnect();
    };

    this.ws.onmessage = (e) => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }

      switch (msg.type) {
        case 'sources':
          if (this.onSources) this.onSources(msg.sources);
          break;
        case 'frame':
          if (this.onFrame) this.onFrame(msg.sourceId, msg.sourceName, msg.jpeg, msg.detections, msg.width, msg.height);
          break;
        case 'detections':
          if (this.onDetections) this.onDetections(msg.sourceId, msg.sourceName, msg.detections);
          break;
      }
    };

    this.ws.onclose = () => {
      this.connected = false;
      if (this.onDisconnect) this.onDisconnect();
      this._scheduleReconnect();
    };

    this.ws.onerror = () => {
      // onclose will fire after this
    };
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._doConnect();
    }, 3000);
  }

  sendFrame(jpegBase64, detections, width, height) {
    if (!this.connected || !this.ws || this.ws.readyState !== 1) return;
    this.ws.send(JSON.stringify({
      type: 'frame',
      jpeg: jpegBase64,
      detections,
      width,
      height,
    }));
  }

  sendDetections(detections) {
    if (!this.connected || !this.ws || this.ws.readyState !== 1) return;
    this.ws.send(JSON.stringify({
      type: 'detections',
      detections,
    }));
  }

  disconnect() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
  }
}
