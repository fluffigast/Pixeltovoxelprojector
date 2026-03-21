import React, { useState, useCallback } from 'react';
import {
  View,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  Text,
  Modal,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import CameraView from '../components/CameraView';
import SettingsPanel from '../components/SettingsPanel';
import StatsPanel from '../components/StatsPanel';
import { Track, Detection, DetectionSettings, DetectionStats } from '../types';

const CameraScreen: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);
  
  const [settings, setSettings] = useState<DetectionSettings>({
    sensitivity: 0.7,
    minArea: 3,
    maxArea: 100,
    confidenceThreshold: 0.6,
    showTrails: true,
    showVegetationZones: false,
    overlayMode: 'analytical',
  });

  const [stats, setStats] = useState<DetectionStats>({
    totalDetections: 0,
    activeTracksCount: 0,
    vegetationZones: 0,
    behaviorCounts: {
      hovering: 0,
      fast: 0,
      normal: 0,
    },
    detectionHistory: [],
  });

  const handleDetectionUpdate = useCallback((newTracks: Track[], newDetections: Detection[]) => {
    setTracks(newTracks);
    setDetections(newDetections);

    // Update stats
    const behaviorCounts = newTracks.reduce(
      (acc, track) => {
        acc[track.behavior] += 1;
        return acc;
      },
      { hovering: 0, fast: 0, normal: 0 }
    );

    setStats(prevStats => ({
      ...prevStats,
      totalDetections: prevStats.totalDetections + newDetections.length,
      activeTracksCount: newTracks.length,
      behaviorCounts,
      detectionHistory: [...prevStats.detectionHistory.slice(-99), newTracks.length], // Keep last 100
    }));
  }, []);

  const handleToggleRecording = useCallback(() => {
    setIsRecording(prev => !prev);
  }, []);

  const handleSettingsChange = useCallback((newSettings: Partial<DetectionSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const resetStats = useCallback(() => {
    setStats({
      totalDetections: 0,
      activeTracksCount: 0,
      vegetationZones: 0,
      behaviorCounts: {
        hovering: 0,
        fast: 0,
        normal: 0,
      },
      detectionHistory: [],
    });
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <CameraView
        settings={settings}
        onDetectionUpdate={handleDetectionUpdate}
        isRecording={isRecording}
        onToggleRecording={handleToggleRecording}
      />

      {/* Top Controls */}
      <View style={styles.topControls}>
        <TouchableOpacity
          style={styles.topButton}
          onPress={() => setShowSettings(true)}
        >
          <Icon name="settings" size={24} color="white" />
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.topButton}
          onPress={() => setShowStats(true)}
        >
          <Icon name="analytics" size={24} color="white" />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.topButton}
          onPress={resetStats}
        >
          <Icon name="refresh" size={24} color="white" />
        </TouchableOpacity>
      </View>

      {/* Quick Settings */}
      <View style={styles.quickSettings}>
        <TouchableOpacity
          style={[styles.quickButton, settings.showTrails && styles.quickButtonActive]}
          onPress={() => handleSettingsChange({ showTrails: !settings.showTrails })}
        >
          <Text style={styles.quickButtonText}>Trails</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.quickButton, settings.showVegetationZones && styles.quickButtonActive]}
          onPress={() => handleSettingsChange({ showVegetationZones: !settings.showVegetationZones })}
        >
          <Text style={styles.quickButtonText}>Zones</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.quickButton}
          onPress={() => {
            const modes: ('minimal' | 'analytical' | 'scientific')[] = ['minimal', 'analytical', 'scientific'];
            const currentIndex = modes.indexOf(settings.overlayMode);
            const nextMode = modes[(currentIndex + 1) % modes.length];
            handleSettingsChange({ overlayMode: nextMode });
          }}
        >
          <Text style={styles.quickButtonText}>
            {settings.overlayMode.charAt(0).toUpperCase() + settings.overlayMode.slice(1)}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Settings Modal */}
      <Modal
        visible={showSettings}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowSettings(false)}
      >
        <SettingsPanel
          settings={settings}
          onSettingsChange={handleSettingsChange}
          onClose={() => setShowSettings(false)}
        />
      </Modal>

      {/* Stats Modal */}
      <Modal
        visible={showStats}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowStats(false)}
      >
        <StatsPanel
          stats={stats}
          tracks={tracks}
          onClose={() => setShowStats(false)}
        />
      </Modal>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  topControls: {
    position: 'absolute',
    top: 10,
    right: 10,
    flexDirection: 'row',
    zIndex: 1000,
  },
  topButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 20,
    padding: 8,
    marginLeft: 8,
  },
  quickSettings: {
    position: 'absolute',
    top: 10,
    left: 10,
    flexDirection: 'column',
    zIndex: 1000,
  },
  quickButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 15,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  quickButtonActive: {
    backgroundColor: 'rgba(0, 255, 100, 0.3)',
    borderColor: '#00FF64',
  },
  quickButtonText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});

export default CameraScreen;