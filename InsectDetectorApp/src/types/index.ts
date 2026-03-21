export interface Detection {
  id: string;
  center: [number, number];
  bbox: [number, number, number, number];
  area: number;
  confidence: number;
  timestamp: number;
  behavior: 'hovering' | 'fast' | 'normal';
  isVegetation: boolean;
  vegetationProbability: number;
}

export interface Track {
  id: string;
  positions: [number, number][];
  lastCenter: [number, number];
  age: number;
  confidence: number;
  behavior: 'hovering' | 'fast' | 'normal';
  speed: number;
  isActive: boolean;
  lastSeen: number;
}

export interface DetectionStats {
  totalDetections: number;
  activeTracksCount: number;
  vegetationZones: number;
  behaviorCounts: {
    hovering: number;
    fast: number;
    normal: number;
  };
  detectionHistory: number[];
}

export interface CameraSettings {
  resolution: 'low' | 'medium' | 'high';
  fps: number;
  enableFlash: boolean;
  enableStabilization: boolean;
}

export interface DetectionSettings {
  sensitivity: number;
  minArea: number;
  maxArea: number;
  confidenceThreshold: number;
  showTrails: boolean;
  showVegetationZones: boolean;
  overlayMode: 'minimal' | 'analytical' | 'scientific';
}