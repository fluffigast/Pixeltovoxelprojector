# Drone & Insect Detector — Design Spec

## Problem
The web frontend uses YOLOv11n (COCO 80-class) which has no drone or insect class. Detection relies on proxies (bird=insect, airplane=drone). The system needs to actually classify flying objects as drone, insect, or bird.

## Approach: Motion-First with Heuristic Classification

### Detection Pipeline
1. **Motion detector (existing):** Three-frame differencing + max-mean spatial filter + connected components + per-blob tracker. This is the primary detector — purpose-built for small point targets (IRST lineage).
2. **Heuristic classifier (new):** Each tracked object is scored over time using observable features to assign a class label.
3. **YOLO (demoted):** Supplementary role only. Used for objects >50px. COCO class mapping: airplane->drone, bird->bird, kite->drone. Other classes ignored unless overlapping a motion blob.

### Heuristic Classification Features

| Feature | Insect | Drone | Bird |
|---------|--------|-------|------|
| Size (px) | 2-30 | 30-300 | 20-200 |
| Speed (px/frame) | high, erratic | moderate, steady | moderate, smooth |
| Trajectory | jittery, random direction changes | linear segments or hovering | smooth curves |
| Track duration | short (flash appearances) | long (persistent) | moderate |

Classification is per-track, not per-frame. Early frames show "unknown", then refine as history accumulates. Minimum 5 frames before assigning a label.

**Jitter score:** Standard deviation of direction changes over last N frames. High jitter = insect. Low jitter + hovering = drone. Low jitter + consistent direction = bird.

**Size consistency:** Drones maintain consistent apparent size. Insects fluctuate. Birds change gradually.

## Multi-Phone Architecture

```
Phone A (camera) --WebSocket--> Server <--WebSocket-- Dashboard
Phone B (camera) --WebSocket-->            (aggregated view)
Phone C (camera) --WebSocket-->
```

### Server (server.js)
- Node.js + `ws` library
- Phones connect as "sources", dashboards connect as "viewers"
- Sources send: compressed JPEG frame + detection array + device metadata
- Server relays to all connected viewers
- No heavy processing on server — detection runs on-device

### Phone Client
- Same web app, opened in mobile browser
- `getUserMedia` for camera access
- Runs motion detection + classification locally
- Streams frames + detections to server via WebSocket
- Sends device info (name, GPS if available)

### Dashboard
- Grid layout of all connected camera feeds
- Combined detection log and stats
- Per-device and aggregate detection counts
- Device status indicators (connected/disconnected/fps)

### Why WebSocket over WebRTC
WebRTC is better for peer-to-peer video but complex for 1-to-many. WebSocket + JPEG is simpler, works everywhere, and detection runs on-device so full-res streaming isn't needed.

## UI Design (BugSpotter-inspired)

### Theme
- Green gradient background (#4CAF50 -> #2E7D32)
- White card-based layout with rounded corners and shadows
- System font stack

### Layout
- Header with title and subtitle
- Source selection: camera, sample videos (insects/drone), file upload, URL input
- Video area: original + detection overlay side by side
- Stats panel: total detections, current frame count, FPS, per-class counts
- Detection log with timestamps
- Settings: motion sensitivity, min/max size, confidence threshold
- Device panel: connected phones with status

### Detection Colors
- Yellow (#FFCC00): insect
- Red (#FF3333): drone
- Cyan (#00DDFF): bird
- Gray (#888888): unknown/unclassified
- Motion-only blobs without classification: dim yellow outline

### Labels
Bounding boxes show: `{class} {confidence}%` for YOLO detections, `{class}` for motion-classified detections.

## File Structure
```
web/
  index.html          -- rebuilt UI
  app.js              -- main app, video sources, inference loop
  detector.js         -- YOLO inference (demoted, supplementary)
  motion.js           -- motion detector (unchanged)
  classifier.js       -- NEW: heuristic classifier
  renderer.js         -- updated colors/labels
  server.js           -- NEW: Node.js WebSocket server
  network.js          -- NEW: client WebSocket module
  package.json        -- NEW: for server dependencies
  models/yolov11n.onnx
  samples/*.mp4
```

## Out of Scope
- Custom trained YOLO model
- GPS/map view
- Session recording/playback
- User authentication
- Species-level identification
