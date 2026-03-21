import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import Svg, { Circle, Rect, Line, Text as SvgText } from 'react-native-svg';
import { Track, Detection, DetectionSettings } from '../types';

interface DetectionOverlayProps {
  tracks: Track[];
  detections: Detection[];
  settings: DetectionSettings;
  screenWidth: number;
  screenHeight: number;
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  tracks,
  detections,
  settings,
  screenWidth,
  screenHeight,
}) => {
  const getColorForBehavior = (behavior: 'hovering' | 'fast' | 'normal'): string => {
    switch (behavior) {
      case 'hovering':
        return '#FFFF00'; // Yellow
      case 'fast':
        return '#FF6400'; // Orange-red
      case 'normal':
      default:
        return '#00FF64'; // Bright green
    }
  };

  const getColorForConfidence = (confidence: number): string => {
    if (confidence > 0.8) return '#FF0000'; // Red - high confidence
    if (confidence > 0.6) return '#FFA500'; // Orange - medium confidence
    return '#FFFF00'; // Yellow - low confidence
  };

  const renderMinimalOverlay = () => {
    return tracks.map((track) => (
      <Circle
        key={track.id}
        cx={track.lastCenter[0]}
        cy={track.lastCenter[1]}
        r={2}
        fill={getColorForBehavior(track.behavior)}
      />
    ));
  };

  const renderAnalyticalOverlay = () => {
    return tracks.map((track) => {
      const color = getColorForBehavior(track.behavior);
      const [x, y] = track.lastCenter;
      
      return (
        <React.Fragment key={track.id}>
          <Circle cx={x} cy={y} r={3} fill={color} />
          <Circle cx={x} cy={y} r={6} stroke={color} strokeWidth={1} fill="none" />
          {track.confidence > 0.8 && (
            <SvgText
              x={x + 8}
              y={y - 8}
              fontSize="10"
              fill={color}
              fontWeight="bold"
            >
              {track.confidence.toFixed(1)}
            </SvgText>
          )}
        </React.Fragment>
      );
    });
  };

  const renderScientificOverlay = () => {
    return tracks.map((track) => {
      const color = getColorForConfidence(track.confidence);
      const [x, y] = track.lastCenter;
      
      // Find corresponding detection for bbox
      const detection = detections.find(d => 
        Math.abs(d.center[0] - x) < 5 && Math.abs(d.center[1] - y) < 5
      );
      
      return (
        <React.Fragment key={track.id}>
          {detection && (
            <Rect
              x={detection.bbox[0]}
              y={detection.bbox[1]}
              width={detection.bbox[2]}
              height={detection.bbox[3]}
              stroke={color}
              strokeWidth={1}
              fill="none"
            />
          )}
          <Circle cx={x} cy={y} r={2} fill={color} />
          <SvgText
            x={detection?.bbox[0] || x}
            y={(detection?.bbox[1] || y) - 5}
            fontSize="8"
            fill={color}
            fontWeight="bold"
          >
            #{track.id.slice(-3)} {track.confidence.toFixed(2)}
          </SvgText>
        </React.Fragment>
      );
    });
  };

  const renderTrails = () => {
    if (!settings.showTrails) return null;

    return tracks.map((track) => {
      if (track.positions.length < 2) return null;

      const color = getColorForBehavior(track.behavior);
      const recentPositions = track.positions.slice(-15); // Last 15 positions

      return (
        <React.Fragment key={`trail-${track.id}`}>
          {recentPositions.slice(0, -1).map((pos, index) => {
            const nextPos = recentPositions[index + 1];
            const alpha = (index + 1) / recentPositions.length; // Fading effect
            
            return (
              <Line
                key={`${track.id}-${index}`}
                x1={pos[0]}
                y1={pos[1]}
                x2={nextPos[0]}
                y2={nextPos[1]}
                stroke={color}
                strokeWidth={1}
                strokeOpacity={alpha}
              />
            );
          })}
        </React.Fragment>
      );
    });
  };

  const renderVegetationZones = () => {
    if (!settings.showVegetationZones) return null;

    // This would need to be connected to the detector's vegetation zones
    // For now, we'll render placeholder zones
    return null;
  };

  const renderOverlayContent = () => {
    switch (settings.overlayMode) {
      case 'minimal':
        return renderMinimalOverlay();
      case 'analytical':
        return renderAnalyticalOverlay();
      case 'scientific':
        return renderScientificOverlay();
      default:
        return renderMinimalOverlay();
    }
  };

  return (
    <View style={styles.container} pointerEvents="none">
      <Svg width={screenWidth} height={screenHeight} style={StyleSheet.absoluteFill}>
        {renderVegetationZones()}
        {renderTrails()}
        {renderOverlayContent()}
      </Svg>
      
      {/* Stats overlay */}
      <View style={styles.statsContainer}>
        <Text style={styles.statsText}>
          Active: {tracks.length} | Total: {detections.length}
        </Text>
        <Text style={styles.statsText}>
          Mode: {settings.overlayMode} | Confidence: {settings.confidenceThreshold.toFixed(1)}
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
  },
  statsContainer: {
    position: 'absolute',
    top: 50,
    left: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 8,
    borderRadius: 5,
  },
  statsText: {
    color: 'white',
    fontSize: 12,
    fontFamily: 'monospace',
  },
});

export default DetectionOverlay;