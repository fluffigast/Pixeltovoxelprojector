import { Detection, Track, DetectionSettings } from '../types';
import { TemporalIntelligence } from './TemporalIntelligence';
import { MovementIntelligence } from './MovementIntelligence';

export class InsectDetector {
  private temporalIntelligence: TemporalIntelligence;
  private movementIntelligence: MovementIntelligence;
  private tracks: Map<string, Track> = new Map();
  private nextTrackId = 0;
  private frameCount = 0;
  private frameBuffer: ImageData[] = [];
  private maxFrameBuffer = 3;

  constructor(private settings: DetectionSettings) {
    this.temporalIntelligence = new TemporalIntelligence();
    this.movementIntelligence = new MovementIntelligence();
  }

  updateSettings(settings: DetectionSettings): void {
    this.settings = settings;
  }

  processFrame(imageData: ImageData): { tracks: Track[], detections: Detection[] } {
    this.frameCount++;
    const currentTime = Date.now();

    // Basic motion detection using frame differencing
    const detections = this.detectMotion(imageData, currentTime);

    // Apply intelligent filtering
    const intelligentDetections = this.applyIntelligentFiltering(detections, currentTime);

    // Update tracks
    const activeTracks = this.updateTracks(intelligentDetections, currentTime);

    // Cleanup old patterns periodically
    if (this.frameCount % 100 === 0) {
      this.temporalIntelligence.cleanupOldPatterns(currentTime);
      this.movementIntelligence.cleanupOldTracks(new Set(Array.from(this.tracks.keys())));
    }

    // Update frame buffer
    this.frameBuffer.push(imageData);
    if (this.frameBuffer.length > this.maxFrameBuffer) {
      this.frameBuffer.shift();
    }

    return { tracks: activeTracks, detections: intelligentDetections };
  }

  private detectMotion(currentImageData: ImageData, timestamp: number): Detection[] {
    if (this.frameBuffer.length === 0) {
      return [];
    }

    const detections: Detection[] = [];
    const currentData = currentImageData.data;
    const prevImageData = this.frameBuffer[this.frameBuffer.length - 1];
    const prevData = prevImageData.data;

    if (currentImageData.width !== prevImageData.width || 
        currentImageData.height !== prevImageData.height) {
      return [];
    }

    const width = currentImageData.width;
    const height = currentImageData.height;

    // Convert to grayscale and detect differences
    const threshold = this.settings.sensitivity * 30; // Adjust threshold based on sensitivity
    const motionPixels: [number, number][] = [];

    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        // Convert to grayscale
        const currentGray = (currentData[idx] + currentData[idx + 1] + currentData[idx + 2]) / 3;
        const prevGray = (prevData[idx] + prevData[idx + 1] + prevData[idx + 2]) / 3;
        
        const diff = Math.abs(currentGray - prevGray);
        
        if (diff > threshold) {
          motionPixels.push([x, y]);
        }
      }
    }

    // Group nearby motion pixels into detections
    const groups = this.groupMotionPixels(motionPixels);
    
    for (const group of groups) {
      const detection = this.analyzeMotionGroup(group, timestamp);
      if (detection) {
        detections.push(detection);
      }
    }

    return detections;
  }

  private groupMotionPixels(pixels: [number, number][]): [number, number][][] {
    const groups: [number, number][][] = [];
    const visited = new Set<string>();
    const maxDistance = 8; // Maximum distance to consider pixels as part of same group

    for (const pixel of pixels) {
      const key = `${pixel[0]},${pixel[1]}`;
      if (visited.has(key)) continue;

      const group: [number, number][] = [];
      const queue: [number, number][] = [pixel];
      
      while (queue.length > 0) {
        const current = queue.shift()!;
        const currentKey = `${current[0]},${current[1]}`;
        
        if (visited.has(currentKey)) continue;
        visited.add(currentKey);
        group.push(current);

        // Find nearby unvisited pixels
        for (const otherPixel of pixels) {
          const otherKey = `${otherPixel[0]},${otherPixel[1]}`;
          if (visited.has(otherKey)) continue;

          const distance = Math.sqrt(
            Math.pow(current[0] - otherPixel[0], 2) + 
            Math.pow(current[1] - otherPixel[1], 2)
          );

          if (distance <= maxDistance) {
            queue.push(otherPixel);
          }
        }
      }

      if (group.length >= 3) { // Minimum group size
        groups.push(group);
      }
    }

    return groups;
  }

  private analyzeMotionGroup(group: [number, number][], timestamp: number): Detection | null {
    if (group.length < 3) return null;

    // Calculate bounding box
    const minX = Math.min(...group.map(p => p[0]));
    const maxX = Math.max(...group.map(p => p[0]));
    const minY = Math.min(...group.map(p => p[1]));
    const maxY = Math.max(...group.map(p => p[1]));

    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    const area = group.length;

    // Filter by size
    if (area < this.settings.minArea || area > this.settings.maxArea) {
      return null;
    }

    // Calculate aspect ratio
    const aspectRatio = width / height;
    if (aspectRatio < 0.1 || aspectRatio > 8.0) {
      return null;
    }

    // Calculate center
    const centerX = Math.floor((minX + maxX) / 2);
    const centerY = Math.floor((minY + maxY) / 2);

    // Basic confidence based on size and compactness
    const boundingBoxArea = width * height;
    const compactness = area / boundingBoxArea;
    let confidence = Math.min(1.0, compactness * 2);

    // Adjust confidence based on size preferences
    if (area >= 5 && area <= 50) {
      confidence += 0.2;
    }

    const detection: Detection = {
      id: `det_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      center: [centerX, centerY],
      bbox: [minX, minY, width, height],
      area,
      confidence,
      timestamp,
      behavior: 'normal',
      isVegetation: false,
      vegetationProbability: 0,
    };

    return detection;
  }

  private applyIntelligentFiltering(detections: Detection[], currentTime: number): Detection[] {
    const filtered: Detection[] = [];

    for (const detection of detections) {
      // Apply temporal intelligence
      const vegetationProbability = this.temporalIntelligence.analyzeDetection(
        detection.center,
        currentTime
      );

      detection.vegetationProbability = vegetationProbability;
      detection.isVegetation = vegetationProbability > 0.6;

      // Filter out likely vegetation if not showing vegetation zones
      if (!detection.isVegetation || this.settings.showVegetationZones) {
        filtered.push(detection);
      }
    }

    return filtered;
  }

  private updateTracks(detections: Detection[], currentTime: number): Track[] {
    const updatedTracks = new Map<string, Track>();
    const assignedDetections = new Set<string>();
    const maxDistance = 50; // Maximum distance for track association

    // Update existing tracks
    for (const [trackId, track] of this.tracks.entries()) {
      let bestDetection: Detection | null = null;
      let bestDistance = Infinity;

      for (const detection of detections) {
        if (assignedDetections.has(detection.id)) continue;

        const distance = Math.sqrt(
          Math.pow(detection.center[0] - track.lastCenter[0], 2) +
          Math.pow(detection.center[1] - track.lastCenter[1], 2)
        );

        if (distance < bestDistance && distance < maxDistance) {
          bestDistance = distance;
          bestDetection = detection;
        }
      }

      if (bestDetection) {
        // Update track
        track.positions.push(bestDetection.center);
        track.lastCenter = bestDetection.center;
        track.age += 1;
        track.lastSeen = currentTime;
        track.isActive = true;

        // Limit position history
        if (track.positions.length > 20) {
          track.positions = track.positions.slice(-20);
        }

        // Analyze movement pattern
        const movementAnalysis = this.movementIntelligence.analyzeTrackMovement(
          trackId,
          bestDetection.center,
          currentTime
        );

        // Calculate behavior and confidence
        this.updateTrackBehavior(track, movementAnalysis, bestDetection);

        updatedTracks.set(trackId, track);
        assignedDetections.add(bestDetection.id);
      } else {
        // Track lost - keep for a few frames
        const timeSinceLastSeen = currentTime - track.lastSeen;
        if (timeSinceLastSeen < 2000) { // 2 seconds
          track.isActive = false;
          updatedTracks.set(trackId, track);
        }
      }
    }

    // Create new tracks for unassigned detections
    for (const detection of detections) {
      if (!assignedDetections.has(detection.id) && !detection.isVegetation) {
        const newTrack: Track = {
          id: `track_${this.nextTrackId++}`,
          positions: [detection.center],
          lastCenter: detection.center,
          age: 1,
          confidence: detection.confidence,
          behavior: 'normal',
          speed: 0,
          isActive: true,
          lastSeen: currentTime,
        };

        updatedTracks.set(newTrack.id, newTrack);
      }
    }

    this.tracks = updatedTracks;

    // Return only active tracks that meet confidence threshold
    return Array.from(this.tracks.values()).filter(
      track => track.isActive && track.confidence >= this.settings.confidenceThreshold
    );
  }

  private updateTrackBehavior(track: Track, movementAnalysis: any, detection: Detection): void {
    if (track.positions.length < 2) return;

    // Calculate speed
    const recentPositions = track.positions.slice(-5);
    let totalDistance = 0;
    
    for (let i = 1; i < recentPositions.length; i++) {
      const dx = recentPositions[i][0] - recentPositions[i-1][0];
      const dy = recentPositions[i][1] - recentPositions[i-1][1];
      totalDistance += Math.sqrt(dx * dx + dy * dy);
    }
    
    track.speed = totalDistance / (recentPositions.length - 1);

    // Determine behavior based on speed and movement patterns
    if (track.speed < 2) {
      track.behavior = 'hovering';
    } else if (track.speed > 15) {
      track.behavior = 'fast';
    } else {
      track.behavior = 'normal';
    }

    // Update confidence based on movement analysis and track age
    const baseConfidence = detection.confidence;
    const movementBonus = movementAnalysis?.isInsectLike || 0.5;
    const ageBonus = Math.min(0.3, track.age * 0.05);
    const vegetationPenalty = detection.vegetationProbability * 0.5;

    track.confidence = Math.max(0.1, Math.min(1.0, 
      baseConfidence + movementBonus * 0.3 + ageBonus - vegetationPenalty
    ));
  }

  getVegetationZones(): Set<string> {
    return this.temporalIntelligence.getVegetationLocations();
  }

  reset(): void {
    this.tracks.clear();
    this.frameBuffer = [];
    this.temporalIntelligence = new TemporalIntelligence();
    this.movementIntelligence = new MovementIntelligence();
    this.frameCount = 0;
    this.nextTrackId = 0;
  }
}