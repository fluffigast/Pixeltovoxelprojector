interface DetectionPattern {
  location: [number, number];
  detectionTimes: number[];
  lastDetection: number;
  totalDetections: number;
  patternScore: number;
  isVegetation: boolean;
  avgInterval: number;
}

export class TemporalIntelligence {
  private gridSize = 20;
  private locationPatterns: Map<string, DetectionPattern> = new Map();
  private vegetationDetectionThreshold = 8;
  private rhythmicThreshold = 0.3;
  private timeWindow = 30000; // 30 seconds in milliseconds
  private vegetationIntervalRange = [500, 3000]; // 0.5-3 seconds in ms
  private vegetationPersistenceThreshold = 0.7;

  analyzeDetection(center: [number, number], timestamp: number): number {
    // Convert to grid coordinates
    const gridX = Math.floor(center[0] / this.gridSize);
    const gridY = Math.floor(center[1] / this.gridSize);
    const gridKey = `${gridX},${gridY}`;

    // Get or create pattern for this location
    let pattern = this.locationPatterns.get(gridKey);
    if (!pattern) {
      pattern = {
        location: [gridX, gridY],
        detectionTimes: [],
        lastDetection: 0,
        totalDetections: 0,
        patternScore: 0,
        isVegetation: false,
        avgInterval: 0,
      };
      this.locationPatterns.set(gridKey, pattern);
    }

    // Update pattern data
    pattern.detectionTimes.push(timestamp);
    pattern.lastDetection = timestamp;
    pattern.totalDetections += 1;

    // Clean old detections outside time window
    const cutoffTime = timestamp - this.timeWindow;
    pattern.detectionTimes = pattern.detectionTimes.filter(t => t >= cutoffTime);

    // Analyze pattern if we have enough data
    if (pattern.detectionTimes.length >= this.vegetationDetectionThreshold) {
      pattern.patternScore = this.calculateVegetationProbability(pattern, timestamp);
      pattern.isVegetation = pattern.patternScore > 0.6;
    }

    return pattern.patternScore;
  }

  private calculateVegetationProbability(pattern: DetectionPattern, currentTime: number): number {
    if (pattern.detectionTimes.length < 3) {
      return 0.0;
    }

    const detectionTimes = pattern.detectionTimes.sort((a, b) => a - b);
    let vegetationScore = 0.0;

    // 1. Analyze timing intervals (vegetation has rhythmic patterns)
    const intervals: number[] = [];
    for (let i = 1; i < detectionTimes.length; i++) {
      intervals.push(detectionTimes[i] - detectionTimes[i - 1]);
    }

    if (intervals.length > 0) {
      const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const variance = intervals.reduce((acc, val) => acc + Math.pow(val - avgInterval, 2), 0) / intervals.length;
      const intervalConsistency = 1.0 / (1.0 + Math.sqrt(variance) / 1000);

      pattern.avgInterval = avgInterval;

      // Vegetation tends to have consistent intervals (wind patterns)
      if (intervalConsistency > 0.5) {
        vegetationScore += 0.3;
      }

      // Check if intervals match vegetation characteristics
      if (avgInterval >= this.vegetationIntervalRange[0] && avgInterval <= this.vegetationIntervalRange[1]) {
        vegetationScore += 0.2;
      }
    }

    // 2. Analyze detection frequency (vegetation appears repeatedly)
    const recentWindow = 10000; // Last 10 seconds
    const recentDetections = detectionTimes.filter(t => currentTime - t <= recentWindow).length;
    const detectionRate = recentDetections / (recentWindow / 1000);

    // High detection rate in same location suggests vegetation
    if (detectionRate > 0.5) {
      vegetationScore += 0.3;
    }

    // 3. Analyze spatial persistence (same exact location)
    const locationConsistency = pattern.detectionTimes.length / pattern.totalDetections;
    if (locationConsistency > this.vegetationPersistenceThreshold) {
      vegetationScore += 0.2;
    }

    // 4. Check for rhythmic patterns (simple periodicity check)
    if (intervals.length >= 10) {
      const periodicityScore = this.analyzePeriodicitySimple(intervals);
      vegetationScore += periodicityScore * 0.2;
    }

    return Math.min(1.0, vegetationScore);
  }

  private analyzePeriodicitySimple(intervals: number[]): number {
    if (intervals.length < 10) {
      return 0.0;
    }

    // Simple autocorrelation approximation
    const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const normalized = intervals.map(x => x - mean);
    
    let maxCorr = 0;
    const maxLag = Math.min(Math.floor(intervals.length / 2), 10);
    
    for (let lag = 1; lag < maxLag; lag++) {
      if (lag >= intervals.length) break;
      
      let correlation = 0;
      let validPairs = 0;
      
      for (let i = 0; i < intervals.length - lag; i++) {
        correlation += normalized[i] * normalized[i + lag];
        validPairs++;
      }
      
      if (validPairs > 0) {
        correlation /= validPairs;
        maxCorr = Math.max(maxCorr, Math.abs(correlation) / 1000); // Normalize
      }
    }

    return Math.min(1.0, maxCorr);
  }

  getVegetationLocations(): Set<string> {
    const vegetationLocations = new Set<string>();
    
    for (const [key, pattern] of this.locationPatterns.entries()) {
      if (pattern.isVegetation) {
        vegetationLocations.add(key);
      }
    }
    
    return vegetationLocations;
  }

  cleanupOldPatterns(currentTime: number): void {
    const cutoffTime = currentTime - this.timeWindow * 2;
    
    for (const [key, pattern] of this.locationPatterns.entries()) {
      if (pattern.lastDetection < cutoffTime) {
        this.locationPatterns.delete(key);
      }
    }
  }
}