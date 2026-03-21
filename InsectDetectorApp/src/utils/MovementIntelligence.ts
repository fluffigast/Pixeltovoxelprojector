interface TrackHistory {
  positions: [number, number][];
  timestamps: number[];
}

interface MovementAnalysis {
  isInsectLike: number;
  movementScore: number;
  totalDisplacement: number;
  randomness: number;
  speedVariation: number;
  directionChanges: number;
  returnTendency: number;
}

export class MovementIntelligence {
  private trackHistory: Map<string, TrackHistory> = new Map();
  private movementPatterns: Map<string, MovementAnalysis> = new Map();
  private maxHistoryLength = 20;

  analyzeTrackMovement(trackId: string, position: [number, number], timestamp: number): MovementAnalysis {
    // Get or create track history
    let history = this.trackHistory.get(trackId);
    if (!history) {
      history = { positions: [], timestamps: [] };
      this.trackHistory.set(trackId, history);
    }

    // Add new position
    history.positions.push(position);
    history.timestamps.push(timestamp);

    // Trim history if too long
    if (history.positions.length > this.maxHistoryLength) {
      history.positions = history.positions.slice(-this.maxHistoryLength);
      history.timestamps = history.timestamps.slice(-this.maxHistoryLength);
    }

    if (history.positions.length < 3) {
      const defaultAnalysis: MovementAnalysis = {
        isInsectLike: 0.5,
        movementScore: 0.5,
        totalDisplacement: 0,
        randomness: 0.5,
        speedVariation: 0,
        directionChanges: 0,
        returnTendency: 0,
      };
      this.movementPatterns.set(trackId, defaultAnalysis);
      return defaultAnalysis;
    }

    // Calculate movement characteristics
    const totalDisplacement = this.calculateTotalDisplacement(history.positions);
    const randomness = this.calculateRandomness(history.positions);
    const speedVariation = this.calculateSpeedVariation(history.positions, history.timestamps);
    const directionChanges = this.calculateDirectionChanges(history.positions);
    const returnTendency = this.calculateReturnTendency(history.positions);

    // Combine scores
    let insectScore = 0.0;

    // High displacement suggests insect
    if (totalDisplacement > 100) {
      insectScore += 0.25;
    } else if (totalDisplacement > 50) {
      insectScore += 0.15;
    }

    // High randomness suggests insect
    insectScore += randomness * 0.2;

    // Speed variation suggests insect
    insectScore += speedVariation * 0.2;

    // Direction changes suggest insect
    insectScore += directionChanges * 0.2;

    // Low return tendency suggests insect (vegetation returns to center)
    insectScore += (1.0 - returnTendency) * 0.15;

    const analysis: MovementAnalysis = {
      isInsectLike: Math.min(1.0, insectScore),
      movementScore: insectScore,
      totalDisplacement,
      randomness,
      speedVariation,
      directionChanges,
      returnTendency,
    };

    this.movementPatterns.set(trackId, analysis);
    return analysis;
  }

  private calculateTotalDisplacement(positions: [number, number][]): number {
    if (positions.length < 2) return 0.0;

    const start = positions[0];
    const end = positions[positions.length - 1];

    return Math.sqrt(Math.pow(end[0] - start[0], 2) + Math.pow(end[1] - start[1], 2));
  }

  private calculateRandomness(positions: [number, number][]): number {
    if (positions.length < 4) return 0.5;

    const n = positions.length;
    
    // Calculate mean position
    const meanX = positions.reduce((sum, pos) => sum + pos[0], 0) / n;
    const meanY = positions.reduce((sum, pos) => sum + pos[1], 0) / n;

    // Calculate deviations from linear trend
    const deviations: number[] = [];
    
    for (let i = 0; i < n; i++) {
      const expectedX = meanX + (i - n/2) * (positions[n-1][0] - positions[0][0]) / n;
      const expectedY = meanY + (i - n/2) * (positions[n-1][1] - positions[0][1]) / n;
      
      const deviation = Math.sqrt(
        Math.pow(positions[i][0] - expectedX, 2) + 
        Math.pow(positions[i][1] - expectedY, 2)
      );
      deviations.push(deviation);
    }

    const avgDeviation = deviations.reduce((a, b) => a + b, 0) / deviations.length;
    
    // Normalize to 0-1 range (higher deviation = more random = more insect-like)
    return Math.min(1.0, avgDeviation / 50.0);
  }

  private calculateSpeedVariation(positions: [number, number][], timestamps: number[]): number {
    if (positions.length < 3) return 0.0;

    const speeds: number[] = [];
    
    for (let i = 1; i < positions.length; i++) {
      const dx = positions[i][0] - positions[i-1][0];
      const dy = positions[i][1] - positions[i-1][1];
      const dt = timestamps[i] - timestamps[i-1];
      
      if (dt > 0) {
        const speed = Math.sqrt(dx*dx + dy*dy) / (dt / 1000); // pixels per second
        speeds.push(speed);
      }
    }

    if (speeds.length < 2) return 0.0;

    const meanSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length;
    const variance = speeds.reduce((acc, val) => acc + Math.pow(val - meanSpeed, 2), 0) / speeds.length;
    const stdDev = Math.sqrt(variance);

    // Coefficient of variation (higher = more varied = more insect-like)
    if (meanSpeed > 0) {
      const cv = stdDev / meanSpeed;
      return Math.min(1.0, cv);
    }

    return 0.0;
  }

  private calculateDirectionChanges(positions: [number, number][]): number {
    if (positions.length < 4) return 0.0;

    let directionChanges = 0;
    let prevDirection: number | null = null;

    for (let i = 2; i < positions.length; i++) {
      // Calculate direction vector
      const dx = positions[i][0] - positions[i-1][0];
      const dy = positions[i][1] - positions[i-1][1];

      if (dx === 0 && dy === 0) continue;

      const currentDirection = Math.atan2(dy, dx);

      if (prevDirection !== null) {
        // Calculate angle difference
        let angleDiff = Math.abs(currentDirection - prevDirection);
        if (angleDiff > Math.PI) {
          angleDiff = 2 * Math.PI - angleDiff;
        }

        // Significant direction change (> 45 degrees)
        if (angleDiff > Math.PI / 4) {
          directionChanges++;
        }
      }

      prevDirection = currentDirection;
    }

    // Normalize by number of possible changes
    const maxChanges = positions.length - 2;
    return maxChanges > 0 ? directionChanges / maxChanges : 0.0;
  }

  private calculateReturnTendency(positions: [number, number][]): number {
    if (positions.length < 5) return 0.0;

    const startPos = positions[0];

    // Calculate distances from start position
    const distancesFromStart: number[] = [];
    for (let i = 1; i < positions.length; i++) {
      const dist = Math.sqrt(
        Math.pow(positions[i][0] - startPos[0], 2) + 
        Math.pow(positions[i][1] - startPos[1], 2)
      );
      distancesFromStart.push(dist);
    }

    // Check if recent positions are closer to start than middle positions
    const recentCount = Math.max(1, Math.floor(distancesFromStart.length * 0.3));
    const middleStart = Math.floor(distancesFromStart.length * 0.3);
    const middleEnd = Math.floor(distancesFromStart.length * 0.7);

    const recentDistances = distancesFromStart.slice(-recentCount);
    const middleDistances = distancesFromStart.slice(middleStart, middleEnd);

    if (middleDistances.length > 0 && recentDistances.length > 0) {
      const avgRecent = recentDistances.reduce((a, b) => a + b, 0) / recentDistances.length;
      const avgMiddle = middleDistances.reduce((a, b) => a + b, 0) / middleDistances.length;

      // If recent positions are closer to start, higher return tendency
      if (avgMiddle > 0) {
        return Math.max(0.0, 1.0 - avgRecent / avgMiddle);
      }
    }

    return 0.0;
  }

  getMovementPattern(trackId: string): MovementAnalysis | null {
    return this.movementPatterns.get(trackId) || null;
  }

  cleanupOldTracks(activeTracks: Set<string>): void {
    // Remove patterns for tracks that are no longer active
    for (const trackId of this.trackHistory.keys()) {
      if (!activeTracks.has(trackId)) {
        this.trackHistory.delete(trackId);
        this.movementPatterns.delete(trackId);
      }
    }
  }
}