import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
  Text,
  Alert,
} from 'react-native';
import {
  Camera,
  useCameraDevices,
  useFrameProcessor,
  Frame,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { Track, Detection, DetectionSettings } from '../types';
import { InsectDetector } from '../utils/InsectDetector';
import DetectionOverlay from './DetectionOverlay';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

interface CameraViewProps {
  settings: DetectionSettings;
  onDetectionUpdate: (tracks: Track[], detections: Detection[]) => void;
  isRecording: boolean;
  onToggleRecording: () => void;
}

const CameraView: React.FC<CameraViewProps> = ({
  settings,
  onDetectionUpdate,
  isRecording,
  onToggleRecording,
}) => {
  const camera = useRef<Camera>(null);
  const devices = useCameraDevices();
  const device = devices.back;

  const [cameraPermission, setCameraPermission] = useState<'authorized' | 'denied' | 'not-determined'>('not-determined');
  const [isActive, setIsActive] = useState(true);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);

  // Initialize detector
  const detectorRef = useRef<InsectDetector>(new InsectDetector(settings));

  useEffect(() => {
    checkCameraPermission();
  }, []);

  useEffect(() => {
    detectorRef.current.updateSettings(settings);
  }, [settings]);

  const checkCameraPermission = async () => {
    const permission = await Camera.getCameraPermissionStatus();
    setCameraPermission(permission);

    if (permission === 'denied') {
      const newPermission = await Camera.requestCameraPermission();
      setCameraPermission(newPermission);
    }
  };

  const frameToImageData = Worklets.createRunInJsFn((frame: Frame): ImageData | null => {
    'worklet';
    
    try {
      // This is a simplified conversion - in a real app, you'd need proper frame conversion
      // For now, we'll create a placeholder ImageData structure
      const width = frame.width;
      const height = frame.height;
      
      // Create mock image data for demonstration
      // In practice, you'd convert the actual frame buffer to ImageData
      const data = new Uint8ClampedArray(width * height * 4);
      
      // Fill with placeholder data (this would be the actual frame conversion)
      for (let i = 0; i < data.length; i += 4) {
        data[i] = 128;     // R
        data[i + 1] = 128; // G
        data[i + 2] = 128; // B
        data[i + 3] = 255; // A
      }
      
      return new ImageData(data, width, height);
    } catch (error) {
      console.warn('Frame conversion failed:', error);
      return null;
    }
  });

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    
    if (!isActive || !isRecording) return;

    try {
      const imageData = frameToImageData(frame);
      if (!imageData) return;

      // Process frame on JS thread
      Worklets.createRunInJsFn((imageData: ImageData) => {
        const detector = detectorRef.current;
        const result = detector.processFrame(imageData);
        
        setTracks(result.tracks);
        setDetections(result.detections);
        onDetectionUpdate(result.tracks, result.detections);
      })(imageData);
      
    } catch (error) {
      console.warn('Frame processing error:', error);
    }
  }, [isActive, isRecording]);

  const handleCameraError = useCallback((error: any) => {
    console.error('Camera error:', error);
    Alert.alert('Camera Error', 'Failed to initialize camera. Please check permissions and try again.');
  }, []);

  if (cameraPermission === 'denied') {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera permission is required</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={checkCameraPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera not available</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        frameProcessor={frameProcessor}
        onError={handleCameraError}
        format={device.formats.find(f => 
          f.videoWidth <= 1920 && 
          f.videoHeight <= 1080 &&
          f.maxFps >= 30
        )}
        fps={30}
        videoStabilizationMode="auto"
        lowLightBoost={false}
      />
      
      <DetectionOverlay
        tracks={tracks}
        detections={detections}
        settings={settings}
        screenWidth={screenWidth}
        screenHeight={screenHeight}
      />

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.recordButton, isRecording && styles.recordButtonActive]}
          onPress={onToggleRecording}
        >
          <Text style={styles.recordButtonText}>
            {isRecording ? 'STOP' : 'START'}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.controlButton}
          onPress={() => setIsActive(!isActive)}
        >
          <Text style={styles.controlButtonText}>
            {isActive ? 'PAUSE' : 'RESUME'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'black',
  },
  permissionText: {
    color: 'white',
    fontSize: 18,
    marginBottom: 20,
    textAlign: 'center',
  },
  permissionButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  controls: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  recordButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderColor: '#FF3B30',
    borderWidth: 2,
    borderRadius: 35,
    width: 70,
    height: 70,
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordButtonActive: {
    backgroundColor: '#FF3B30',
  },
  recordButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  controlButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderColor: 'white',
    borderWidth: 1,
    borderRadius: 25,
    paddingHorizontal: 15,
    paddingVertical: 10,
  },
  controlButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
});

export default CameraView;