#!/usr/bin/env python3
"""Advanced insect detection with multi-layer filtering and AI-enhanced tracking."""

import cv2
import numpy as np
import time
import subprocess
import argparse
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from scipy.spatial.distance import cdist

class MultiLayerInsectDetector:
    """Advanced multi-layer insect detection system."""
    
    def __init__(self):
        # Ultra-precise size parameters
        self.min_area = 2           # Even smaller - single pixels
        self.max_area = 120         # Slightly larger for moths
        self.min_perimeter = 4      # Minimum outline
        self.max_perimeter = 80     # Maximum outline
        
        # Shape analysis
        self.min_aspect_ratio = 0.1
        self.max_aspect_ratio = 8.0
        self.min_solidity = 0.2     # Compactness
        self.max_solidity = 1.0
        self.min_extent = 0.1       # Bounding box fill
        
        # Motion parameters
        self.motion_threshold = 5   # Ultra-sensitive
        self.adaptive_threshold = True
        self.gaussian_blur = 1      # Minimal blur
        
        # Multi-layer background subtraction
        self.bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=4, detectShadows=False
        )
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=100, dist2Threshold=200, detectShadows=False
        )
        
        # Frame differencing with multiple scales
        self.frame_buffer = deque(maxlen=5)
        self.pyramid_levels = 3
        
        # Advanced tracking
        self.tracks = {}
        self.next_track_id = 0
        self.kalman_filters = {}
        
        # Behavioral analysis
        self.speed_history_length = 10
        self.direction_history_length = 8
        self.hover_detection_frames = 15
        self.zigzag_angle_threshold = 30
        
        # Machine learning features
        self.feature_history = defaultdict(lambda: deque(maxlen=20))
        
        # Adaptive parameters
        self.auto_tune = True
        self.detection_rate_target = 5  # Target detections per second
        self.last_detection_count = deque(maxlen=30)
        
    def create_kalman_filter(self):
        """Create Kalman filter for smooth tracking."""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        return kalman
        
    def detect_insects_multilayer(self, frame: np.ndarray) -> List[Dict]:
        """Multi-layer detection combining multiple techniques."""
        detections = []
        
        # Layer 1: Enhanced background subtraction
        bg_detections = self._detect_background_subtraction_enhanced(frame)
        detections.extend(bg_detections)
        
        # Layer 2: Multi-scale frame differencing
        diff_detections = self._detect_multiscale_difference(frame)
        detections.extend(diff_detections)
        
        # Layer 3: Optical flow for small objects
        flow_detections = self._detect_optical_flow(frame)
        detections.extend(flow_detections)
        
        # Layer 4: Texture-based detection
        texture_detections = self._detect_texture_anomalies(frame)
        detections.extend(texture_detections)
        
        # Add frame to buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        # Fusion and filtering
        fused_detections = self._fuse_and_filter_detections(detections, frame)
        
        # Adaptive parameter tuning
        if self.auto_tune:
            self._adapt_parameters(len(fused_detections))
            
        return fused_detections
    
    def _detect_background_subtraction_enhanced(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced background subtraction with dual algorithms."""
        detections = []
        
        # MOG2 background subtraction
        fg_mask_mog2 = self.bg_subtractor_mog2.apply(frame)
        
        # KNN background subtraction  
        fg_mask_knn = self.bg_subtractor_knn.apply(frame)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(fg_mask_mog2, fg_mask_knn)
        
        # Enhance small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._analyze_contour_advanced(contour, 'background')
            if detection:
                detections.append(detection)
                
        return detections
    
    def _detect_multiscale_difference(self, frame: np.ndarray) -> List[Dict]:
        """Multi-scale frame differencing for different insect sizes."""
        if len(self.frame_buffer) < 2:
            return []
            
        detections = []
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create image pyramid
        current_pyramid = self._create_pyramid(current_gray, self.pyramid_levels)
        
        for prev_gray in list(self.frame_buffer)[-2:]:
            prev_pyramid = self._create_pyramid(prev_gray, self.pyramid_levels)
            
            # Process each scale
            for level in range(self.pyramid_levels):
                scale_factor = 2 ** level
                diff = cv2.absdiff(current_pyramid[level], prev_pyramid[level])
                
                # Adaptive threshold based on image statistics
                threshold = max(self.motion_threshold, np.std(diff) * 0.5)
                _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                
                # Find contours at this scale
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Scale contour back to original size
                    scaled_contour = contour * scale_factor
                    detection = self._analyze_contour_advanced(scaled_contour, f'difference_scale_{level}')
                    if detection:
                        detections.append(detection)
        
        return detections
    
    def _detect_optical_flow(self, frame: np.ndarray) -> List[Dict]:
        """Optical flow detection for small moving objects."""
        if len(self.frame_buffer) < 1:
            return []
            
        detections = []
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = self.frame_buffer[-1]
        
        # Dense optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, current_gray, None, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Calculate flow magnitude
        if flow[0] is not None:
            flow_mag = np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2)
            
            # Threshold flow magnitude for small, fast movements
            flow_thresh = flow_mag > 2.0  # Pixels moving > 2 pixels/frame
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(flow_thresh.astype(np.uint8))
            
            for label in range(1, num_labels):
                mask = (labels == label)
                if np.sum(mask) >= self.min_area:
                    # Create contour from mask
                    coords = np.column_stack(np.where(mask))
                    if len(coords) > 0:
                        # Convert to contour format
                        contour = np.array([(y, x) for x, y in coords]).reshape(-1, 1, 2)
                        detection = self._analyze_contour_advanced(contour, 'optical_flow')
                        if detection:
                            # Add flow information
                            detection['flow_magnitude'] = np.mean(flow_mag[mask])
                            detections.append(detection)
        
        return detections
    
    def _detect_texture_anomalies(self, frame: np.ndarray) -> List[Dict]:
        """Detect insects based on texture anomalies."""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern for texture analysis
        lbp = self._local_binary_pattern(gray, 8, 1)
        
        # Calculate local texture variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(lbp, -1, kernel)
        local_variance = cv2.filter2D((lbp - local_mean)**2, -1, kernel)
        
        # Find areas with high texture variance (potential insects)
        variance_threshold = np.mean(local_variance) + 2 * np.std(local_variance)
        texture_mask = local_variance > variance_threshold
        
        # Find contours in texture mask
        contours, _ = cv2.findContours(texture_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._analyze_contour_advanced(contour, 'texture')
            if detection:
                detection['texture_variance'] = np.mean(local_variance[texture_mask])
                detections.append(detection)
        
        return detections
    
    def _local_binary_pattern(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        lbp = np.zeros_like(image)
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Bilinear interpolation
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + 1, y1 + 1
            
            # Create shifted images
            shifted = np.roll(np.roll(image, int(y), axis=0), int(x), axis=1)
            lbp += (shifted >= image) * (2 ** i)
        
        return lbp
    
    def _create_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """Create image pyramid for multi-scale processing."""
        pyramid = [image]
        for i in range(1, levels):
            scaled = cv2.pyrDown(pyramid[-1])
            pyramid.append(scaled)
        return pyramid
    
    def _analyze_contour_advanced(self, contour, method: str) -> Optional[Dict]:
        """Advanced contour analysis with multiple geometric features."""
        if len(contour) < 5:
            return None
            
        # Basic measurements
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter < self.min_perimeter or perimeter > self.max_perimeter:
            return None
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        
        # Geometric features
        aspect_ratio = w / h if h > 0 else 0
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return None
            
        # Solidity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if not (self.min_solidity <= solidity <= self.max_solidity):
            return None
            
        # Extent (bounding box ratio)
        extent = area / (w * h) if w * h > 0 else 0
        if extent < self.min_extent:
            return None
        
        # Additional geometric features
        # Compactness
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Eccentricity (ellipse fit)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2) if major_axis > 0 else 0
        else:
            eccentricity = 0
            
        # Roundness
        roundness = 4 * area / (np.pi * perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'center': (center_x, center_y),
            'bbox': (x, y, w, h),
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent,
            'compactness': compactness,
            'eccentricity': eccentricity,
            'roundness': roundness,
            'contour': contour,
            'method': method
        }
    
    def _fuse_and_filter_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Fuse overlapping detections and apply advanced filtering."""
        if not detections:
            return []
        
        # Group nearby detections
        centers = np.array([det['center'] for det in detections])
        distances = cdist(centers, centers)
        
        # Merge nearby detections
        merged_detections = []
        used_indices = set()
        
        for i, detection in enumerate(detections):
            if i in used_indices:
                continue
                
            # Find nearby detections
            nearby_indices = np.where(distances[i] < 15)[0]
            nearby_detections = [detections[j] for j in nearby_indices if j not in used_indices]
            
            if len(nearby_detections) > 1:
                # Merge detections
                merged = self._merge_detections(nearby_detections)
                merged_detections.append(merged)
                used_indices.update(nearby_indices)
            else:
                merged_detections.append(detection)
                used_indices.add(i)
        
        # Apply advanced filtering
        filtered_detections = []
        for detection in merged_detections:
            if self._passes_advanced_filter(detection, frame):
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _merge_detections(self, detections: List[Dict]) -> Dict:
        """Merge multiple overlapping detections."""
        # Calculate weighted center based on area
        total_area = sum(det['area'] for det in detections)
        center_x = sum(det['center'][0] * det['area'] for det in detections) / total_area
        center_y = sum(det['center'][1] * det['area'] for det in detections) / total_area
        
        # Take best features from largest detection
        largest_det = max(detections, key=lambda d: d['area'])
        
        merged = largest_det.copy()
        merged['center'] = (int(center_x), int(center_y))
        merged['area'] = total_area
        merged['method'] = 'merged'
        merged['merge_count'] = len(detections)
        
        return merged
    
    def _passes_advanced_filter(self, detection: Dict, frame: np.ndarray) -> bool:
        """Advanced filtering based on multiple criteria."""
        # Size filtering
        area = detection['area']
        if area < 3 or area > 100:
            return False
            
        # Shape filtering
        if detection['compactness'] < 0.1 or detection['compactness'] > 1.0:
            return False
            
        if detection['solidity'] < 0.2:
            return False
            
        # Intensity analysis
        center_x, center_y = detection['center']
        x, y, w, h = detection['bbox']
        
        # Check if detection is in valid image area
        if (x < 0 or y < 0 or x + w >= frame.shape[1] or y + h >= frame.shape[0]):
            return False
            
        # Analyze pixel intensities in detection area
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return False
            
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        intensity_var = np.var(gray_roi)
        
        # Insects typically have some intensity variation
        if intensity_var < 5:  # Too uniform, likely noise
            return False
            
        # Edge density (insects have edges)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.02:  # Too few edges
            return False
            
        return True
    
    def _adapt_parameters(self, current_detections: int):
        """Adaptively tune parameters based on detection rate."""
        self.last_detection_count.append(current_detections)
        
        if len(self.last_detection_count) >= 30:
            avg_detections = np.mean(self.last_detection_count)
            
            # Too many detections - increase selectivity
            if avg_detections > self.detection_rate_target * 2:
                self.motion_threshold = min(20, self.motion_threshold + 1)
                self.min_area = min(10, self.min_area + 1)
                
            # Too few detections - increase sensitivity
            elif avg_detections < self.detection_rate_target * 0.5:
                self.motion_threshold = max(3, self.motion_threshold - 1)
                self.min_area = max(2, self.min_area - 1)
    
    def update_tracks_advanced(self, detections: List[Dict]) -> List[Dict]:
        """Advanced tracking with Kalman filtering and behavioral analysis."""
        if not detections:
            # Update all tracks with no detections
            for track_id in list(self.tracks.keys()):
                track = self.tracks[track_id]
                track['missed_frames'] += 1
                if track['missed_frames'] > 10:
                    del self.tracks[track_id]
                    if track_id in self.kalman_filters:
                        del self.kalman_filters[track_id]
            return []
        
        # Predict next positions using Kalman filters
        predictions = {}
        for track_id, kalman in self.kalman_filters.items():
            if track_id in self.tracks:
                prediction = kalman.predict()
                predictions[track_id] = (int(prediction[0]), int(prediction[1]))
        
        # Associate detections with tracks
        detection_centers = np.array([det['center'] for det in detections])
        updated_tracks = {}
        assigned_detections = set()
        
        # Update existing tracks
        for track_id, track in self.tracks.items():
            best_detection_idx = None
            best_distance = float('inf')
            
            # Use Kalman prediction if available
            if track_id in predictions:
                predicted_pos = predictions[track_id]
            else:
                predicted_pos = track['last_center']
            
            # Find closest detection
            for i, center in enumerate(detection_centers):
                if i in assigned_detections:
                    continue
                    
                distance = np.sqrt((center[0] - predicted_pos[0])**2 + 
                                 (center[1] - predicted_pos[1])**2)
                
                # Dynamic association distance based on track speed
                max_distance = min(50, max(20, track.get('speed', 10) * 3))
                
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            if best_detection_idx is not None:
                # Update track
                detection = detections[best_detection_idx]
                track['positions'].append(detection['center'])
                track['last_center'] = detection['center']
                track['missed_frames'] = 0
                track['detection'] = detection
                
                # Update Kalman filter
                if track_id in self.kalman_filters:
                    kalman = self.kalman_filters[track_id]
                    measurement = np.array([[np.float32(detection['center'][0])], 
                                          [np.float32(detection['center'][1])]])
                    kalman.correct(measurement)
                
                # Advanced behavioral analysis
                self._analyze_advanced_behavior(track)
                
                updated_tracks[track_id] = track
                assigned_detections.add(best_detection_idx)
            else:
                # Track missed
                track['missed_frames'] += 1
                if track['missed_frames'] <= 5:  # Keep for a few frames
                    updated_tracks[track_id] = track
                else:
                    # Remove track and its Kalman filter
                    if track_id in self.kalman_filters:
                        del self.kalman_filters[track_id]
        
        # Create new tracks for unassigned detections
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                track_id = self.next_track_id
                
                new_track = {
                    'id': track_id,
                    'positions': [detection['center']],
                    'last_center': detection['center'],
                    'missed_frames': 0,
                    'detection': detection,
                    'speed': 0,
                    'acceleration': 0,
                    'direction_changes': 0,
                    'avg_direction': 0,
                    'is_hovering': False,
                    'is_zigzag': False,
                    'is_accelerating': False,
                    'confidence': 0.3,
                    'age': 0
                }
                
                # Initialize Kalman filter
                kalman = self.create_kalman_filter()
                kalman.statePre = np.array([detection['center'][0], detection['center'][1], 0, 0], dtype=np.float32)
                kalman.statePost = np.array([detection['center'][0], detection['center'][1], 0, 0], dtype=np.float32)
                
                self.kalman_filters[track_id] = kalman
                updated_tracks[track_id] = new_track
                self.next_track_id += 1
        
        self.tracks = updated_tracks
        
        # Return tracks with high confidence
        return [track for track in self.tracks.values() 
                if track['missed_frames'] == 0 and track['confidence'] > 0.4]
    
    def _analyze_advanced_behavior(self, track: Dict):
        """Advanced behavioral analysis for insect classification."""
        positions = track['positions']
        track['age'] += 1
        
        if len(positions) < 2:
            return
        
        # Calculate instantaneous speed and acceleration
        recent_positions = positions[-min(self.speed_history_length, len(positions)):]
        speeds = []
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            speed = np.sqrt(dx*dx + dy*dy)
            speeds.append(speed)
        
        if speeds:
            track['speed'] = np.mean(speeds)
            
            # Calculate acceleration
            if len(speeds) > 1:
                accelerations = [speeds[i] - speeds[i-1] for i in range(1, len(speeds))]
                track['acceleration'] = np.mean(np.abs(accelerations))
                track['is_accelerating'] = track['acceleration'] > 2.0
        
        # Direction analysis
        if len(positions) >= 3:
            directions = []
            recent_pos = positions[-min(self.direction_history_length, len(positions)):]
            
            for i in range(1, len(recent_pos)):
                dx = recent_pos[i][0] - recent_pos[i-1][0]
                dy = recent_pos[i][1] - recent_pos[i-1][1]
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)
            
            if len(directions) > 1:
                # Count sharp direction changes (zigzag behavior)
                direction_changes = 0
                for i in range(1, len(directions)):
                    angle_diff = np.abs(directions[i] - directions[i-1])
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    if np.degrees(angle_diff) > self.zigzag_angle_threshold:
                        direction_changes += 1
                
                track['direction_changes'] = direction_changes
                track['is_zigzag'] = direction_changes > len(directions) * 0.3
                
                # Average direction
                track['avg_direction'] = np.mean(directions)
        
        # Hovering detection
        if len(positions) >= self.hover_detection_frames:
            recent_pos = positions[-self.hover_detection_frames:]
            center_x = np.mean([pos[0] for pos in recent_pos])
            center_y = np.mean([pos[1] for pos in recent_pos])
            
            max_distance = max(np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
                             for pos in recent_pos)
            track['is_hovering'] = max_distance < 8
        
        # Update confidence based on insect-like features
        confidence = 0.3
        
        # Size bonus (smaller objects more likely to be insects)
        area = track['detection']['area']
        if 3 <= area <= 30:
            confidence += 0.3
        elif 30 < area <= 60:
            confidence += 0.2
        
        # Movement pattern bonuses
        if 2 <= track['speed'] <= 40:
            confidence += 0.2
            
        if track['is_zigzag']:
            confidence += 0.2
            
        if track['is_hovering']:
            confidence += 0.15
            
        if track['is_accelerating']:
            confidence += 0.1
        
        # Track age bonus (longer tracks more reliable)
        if track['age'] > 10:
            confidence += 0.1
        elif track['age'] > 5:
            confidence += 0.05
        
        # Shape quality bonus
        detection = track['detection']
        if detection.get('compactness', 0) > 0.3:
            confidence += 0.1
            
        # Method bonus (some methods more reliable)
        if detection.get('method') == 'merged':
            confidence += 0.1
        
        track['confidence'] = min(1.0, confidence)


class AdvancedInsectVoxelTracker:
    """Advanced 3D voxel tracker with machine learning features."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.detector = MultiLayerInsectDetector()
        
        # Video parameters
        self.cap = None
        self.width = 800  # Higher resolution for better detection
        self.height = 600
        
        # High-resolution voxel system
        self.voxel_dims = (120, 120, 60)
        self.voxel_size = 0.015  # Even smaller voxels
        self.insect_voxels = np.zeros(self.voxel_dims, dtype=np.float32)
        self.voxel_colors = np.zeros((*self.voxel_dims, 3), dtype=np.float32)
        self.voxel_velocities = np.zeros((*self.voxel_dims, 3), dtype=np.float32)
        self.voxel_decay = 0.94
        
        # Camera model
        self.camera_fov = 75
        self.camera_z = 0.8
        
        # Visualization modes
        self.show_all_detections = False
        self.show_confidence_only = True
        self.min_confidence_display = 0.6
        self.show_behavior_labels = True
        self.show_prediction = True
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)
        self.detection_stats = defaultdict(int)
        
    def get_stream_url(self):
        """Get stream URL with better quality."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            try:
                cmd = ['yt-dlp', '-f', 'best[height<=1080]', '--get-url', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
        
    def start(self):
        """Start advanced insect detection."""
        url = self.get_stream_url()
        print(f"Starting Advanced Insect Detection: {url}")
        
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
            
        # Set higher quality capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        print("🔬 Advanced Multi-Layer Insect Detection Started!")
        print("\nFeatures:")
        print("  ✓ Multi-layer detection (4 algorithms)")
        print("  ✓ Kalman filter tracking")
        print("  ✓ Behavioral analysis")
        print("  ✓ Adaptive parameters")
        print("  ✓ High-resolution voxels")
        print("\nAdvanced Controls:")
        print("  q/ESC - Quit")
        print("  r - Reset system")
        print("  a - Toggle all detections")
        print("  b - Toggle behavior labels")
        print("  p - Toggle prediction")
        print("  = - Lower confidence threshold")
        print("  - - Raise confidence threshold")
        print("  t - Toggle auto-tuning")
        print("  s - Manual sensitivity up")
        print("  d - Manual sensitivity down")
        print("  h - Show help")
        
        cv2.namedWindow('Advanced Insect Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Advanced Insect Detector', 1200, 800)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize for processing while maintaining aspect ratio
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Multi-layer detection
            detections = self.detector.detect_insects_multilayer(frame)
            
            # Advanced tracking
            active_tracks = self.detector.update_tracks_advanced(detections)
            
            # Update voxels
            confident_tracks = [t for t in active_tracks if t['confidence'] >= self.min_confidence_display]
            self._update_advanced_voxels(confident_tracks)
            
            # Create advanced visualization
            vis_frame = self._create_advanced_visualization(frame, detections, active_tracks)
            
            cv2.imshow('Advanced Insect Detector', vis_frame)
            
            # Performance tracking
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            
            # Update statistics
            for detection in detections:
                self.detection_stats[detection.get('method', 'unknown')] += 1
            
            # Handle advanced controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r') or key == ord('R'):
                self._reset_advanced_system()
            elif key == ord('a') or key == ord('A'):
                self.show_all_detections = not self.show_all_detections
                print(f"Show all detections: {self.show_all_detections}")
            elif key == ord('b') or key == ord('B'):
                self.show_behavior_labels = not self.show_behavior_labels
                print(f"Behavior labels: {self.show_behavior_labels}")
            elif key == ord('p') or key == ord('P'):
                self.show_prediction = not self.show_prediction
                print(f"Show prediction: {self.show_prediction}")
            elif key == ord('=') or key == ord('+'):
                self.min_confidence_display = max(0.1, self.min_confidence_display - 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('-') or key == ord('_'):
                self.min_confidence_display = min(0.9, self.min_confidence_display + 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('t') or key == ord('T'):
                self.detector.auto_tune = not self.detector.auto_tune
                print(f"Auto-tuning: {self.detector.auto_tune}")
            elif key == ord('s') or key == ord('S'):
                self.detector.motion_threshold = max(2, self.detector.motion_threshold - 1)
                print(f"Motion sensitivity: {self.detector.motion_threshold}")
            elif key == ord('d') or key == ord('D'):
                self.detector.motion_threshold = min(30, self.detector.motion_threshold + 1)
                print(f"Motion sensitivity: {self.detector.motion_threshold}")
            elif key == ord('h') or key == ord('H'):
                self._show_advanced_help()
                
        # Show final statistics
        elapsed = time.time() - start_time
        print(f"\nSession Statistics:")
        print(f"  Runtime: {elapsed:.1f} seconds")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {frame_count / elapsed:.1f}")
        print(f"  Detection methods:")
        for method, count in self.detection_stats.items():
            print(f"    {method}: {count}")
                
    def _update_advanced_voxels(self, tracks: List[Dict]):
        """Update voxels with advanced features."""
        # Apply decay
        self.insect_voxels *= self.voxel_decay
        self.voxel_colors *= self.voxel_decay
        self.voxel_velocities *= self.voxel_decay
        
        for track in tracks:
            center_x, center_y = track['last_center']
            
            # Advanced depth estimation
            area = track['detection']['area']
            speed = track['speed']
            confidence = track['confidence']
            
            # Multi-factor depth estimation
            size_depth = max(0.1, min(1.2, 40.0 / (area + 5)))
            speed_depth = max(0.8, min(1.5, speed / 20.0))
            estimated_depth = (size_depth + speed_depth) / 2.0
            
            # Convert to 3D coordinates
            norm_x = (center_x / self.width - 0.5) * 2.0
            norm_y = (center_y / self.height - 0.5) * 2.0
            
            fov_scale = math.tan(math.radians(self.camera_fov / 2))
            world_x = norm_x * fov_scale * estimated_depth
            world_y = -norm_y * fov_scale * estimated_depth
            world_z = self.camera_z + estimated_depth
            
            # Velocity estimation
            if len(track['positions']) >= 2:
                prev_pos = track['positions'][-2]
                velocity_x = (center_x - prev_pos[0]) * 0.1
                velocity_y = (center_y - prev_pos[1]) * 0.1
                velocity_z = 0
            else:
                velocity_x = velocity_y = velocity_z = 0
            
            # Convert to voxel coordinates
            vx = int(world_x / self.voxel_size + self.voxel_dims[0] / 2)
            vy = int(world_y / self.voxel_size + self.voxel_dims[1] / 2)
            vz = int(world_z / self.voxel_size)
            
            if (0 <= vx < self.voxel_dims[0] and 
                0 <= vy < self.voxel_dims[1] and 
                0 <= vz < self.voxel_dims[2]):
                
                # Advanced color coding
                if track['is_hovering']:
                    color = np.array([1.0, 1.0, 0.0])  # Yellow - hovering
                elif track['is_zigzag']:
                    color = np.array([1.0, 0.0, 1.0])  # Magenta - zigzag
                elif track['is_accelerating']:
                    color = np.array([0.0, 1.0, 1.0])  # Cyan - accelerating
                elif confidence > 0.8:
                    color = np.array([1.0, 0.0, 0.0])  # Red - high confidence
                else:
                    color = np.array([0.0, 1.0, 0.0])  # Green - normal
                
                intensity = confidence * 1.5
                
                # Update voxels with spreading based on confidence
                spread = max(1, int(confidence * 2))
                for dx in range(-spread, spread + 1):
                    for dy in range(-spread, spread + 1):
                        for dz in range(-1, 2):
                            nvx, nvy, nvz = vx + dx, vy + dy, vz + dz
                            if (0 <= nvx < self.voxel_dims[0] and 
                                0 <= nvy < self.voxel_dims[1] and 
                                0 <= nvz < self.voxel_dims[2]):
                                
                                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                                falloff = max(0.1, 1.0 / (1.0 + distance))
                                
                                update_amount = intensity * falloff
                                self.insect_voxels[nvx, nvy, nvz] += update_amount
                                self.voxel_colors[nvx, nvy, nvz] += color * update_amount
                                
                                # Update velocity
                                self.voxel_velocities[nvx, nvy, nvz, 0] += velocity_x * falloff
                                self.voxel_velocities[nvx, nvy, nvz, 1] += velocity_y * falloff
                                self.voxel_velocities[nvx, nvy, nvz, 2] += velocity_z * falloff
                                
    def _create_advanced_visualization(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]) -> np.ndarray:
        """Create advanced visualization with all features."""
        vis = frame.copy()
        
        # Filter tracks for display
        if self.show_confidence_only:
            display_tracks = [t for t in tracks if t['confidence'] >= self.min_confidence_display]
        else:
            display_tracks = tracks if self.show_all_detections else [t for t in tracks if t['confidence'] >= 0.5]
        
        # Draw detection boxes with advanced info
        for track in display_tracks:
            detection = track['detection']
            x, y, w, h = detection['bbox']
            center_x, center_y = track['last_center']
            confidence = track['confidence']
            
            # Dynamic color based on multiple factors
            if confidence > 0.9:
                color = (0, 0, 255)    # Red - very high confidence
            elif track['is_hovering']:
                color = (0, 255, 255)  # Yellow - hovering
            elif track['is_zigzag']:
                color = (255, 0, 255)  # Magenta - zigzag
            elif track['is_accelerating']:
                color = (255, 255, 0)  # Cyan - accelerating
            elif confidence > 0.7:
                color = (0, 165, 255)  # Orange - good confidence
            else:
                color = (0, 255, 0)    # Green - possible
            
            # Draw detection with thickness based on confidence
            thickness = max(1, int(confidence * 3))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            cv2.circle(vis, (center_x, center_y), 3, color, -1)
            
            # Advanced labeling
            labels = []
            labels.append(f"{confidence:.2f}")
            
            if self.show_behavior_labels:
                if track['is_hovering']:
                    labels.append("HOVER")
                if track['is_zigzag']:
                    labels.append("ZIGZAG")
                if track['is_accelerating']:
                    labels.append("ACCEL")
                    
            label_text = " ".join(labels)
            cv2.putText(vis, label_text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Show track ID for high confidence
            if confidence > 0.8:
                cv2.putText(vis, f"ID:{track['id']}", (x + w + 2, y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Prediction visualization
            if self.show_prediction and track['id'] in self.detector.kalman_filters:
                kalman = self.detector.kalman_filters[track['id']]
                predicted = kalman.statePre
                pred_x, pred_y = int(predicted[0]), int(predicted[1])
                
                # Draw prediction
                cv2.circle(vis, (pred_x, pred_y), 5, (255, 255, 255), 1)
                cv2.line(vis, (center_x, center_y), (pred_x, pred_y), (255, 255, 255), 1)
        
        # Draw advanced track trails
        for track in display_tracks:
            if track['confidence'] > 0.6:
                positions = track['positions'][-15:]
                if len(positions) > 1:
                    # Color trail based on behavior
                    if track['is_hovering']:
                        trail_color = (128, 255, 255)
                    elif track['is_zigzag']:
                        trail_color = (255, 128, 255)
                    else:
                        trail_color = (255, 255, 128)
                        
                    pts = np.array(positions, np.int32)
                    cv2.polylines(vis, [pts], False, trail_color, 2)
        
        # Draw voxel overlay
        self._draw_advanced_voxel_overlay(vis)
        
        # Add comprehensive UI
        self._add_advanced_ui(vis, detections, tracks)
        
        return vis
        
    def _draw_advanced_voxel_overlay(self, frame: np.ndarray):
        """Draw advanced voxel overlay with velocity vectors."""
        active_mask = self.insect_voxels > 0.03
        
        if not np.any(active_mask):
            return
            
        coords = np.argwhere(active_mask)
        intensities = self.insect_voxels[active_mask]
        colors = self.voxel_colors[active_mask]
        velocities = self.voxel_velocities[active_mask]
        
        # Normalize colors
        colors = colors / (intensities[:, np.newaxis] + 0.001)
        colors = np.clip(colors, 0, 1)
        
        # Limit for performance
        if len(coords) > 2000:
            indices = np.argsort(intensities)[-2000:]
            coords = coords[indices]
            intensities = intensities[indices]
            colors = colors[indices]
            velocities = velocities[indices]
        
        for i, (coord, intensity, color, velocity) in enumerate(zip(coords, intensities, colors, velocities)):
            # Project to screen
            world_x = (coord[0] - self.voxel_dims[0] / 2) * self.voxel_size
            world_y = (coord[1] - self.voxel_dims[1] / 2) * self.voxel_size
            world_z = coord[2] * self.voxel_size
            
            if world_z > 0.1:
                distance = world_z - self.camera_z
                if distance > 0:
                    fov_scale = math.tan(math.radians(self.camera_fov / 2))
                    screen_x = int((world_x / (distance * fov_scale) + 1) * self.width / 2)
                    screen_y = int((-world_y / (distance * fov_scale) + 1) * self.height / 2)
                    
                    if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                        # Draw voxel with size based on intensity and distance
                        size = max(1, int(6 * intensity / distance))
                        bgr_color = tuple(map(int, color[::-1] * 255))
                        
                        # Glow effect
                        for r in range(size + 2, 0, -1):
                            alpha = 0.3 * (r / (size + 2))
                            glow_color = tuple(int(c * alpha) for c in bgr_color)
                            cv2.circle(frame, (screen_x, screen_y), r, glow_color, -1)
                        
                        # Bright center
                        cv2.circle(frame, (screen_x, screen_y), size, bgr_color, -1)
                        
                        # Draw velocity vector
                        if np.linalg.norm(velocity[:2]) > 0.1:
                            vel_end_x = screen_x + int(velocity[0] * 10)
                            vel_end_y = screen_y + int(velocity[1] * 10)
                            cv2.arrowedLine(frame, (screen_x, screen_y), (vel_end_x, vel_end_y), 
                                          (255, 255, 255), 1, tipLength=0.3)
                            
    def _add_advanced_ui(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]):
        """Add comprehensive UI information."""
        # Performance metrics
        if self.frame_times:
            fps = 1.0 / np.mean(self.frame_times)
            fps_color = (0, 255, 0) if fps > 15 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        else:
            fps = 0
            fps_color = (0, 0, 255)
        
        # Count by confidence levels
        very_high = sum(1 for t in tracks if t['confidence'] > 0.9)
        high = sum(1 for t in tracks if 0.8 <= t['confidence'] <= 0.9)
        medium = sum(1 for t in tracks if 0.6 <= t['confidence'] < 0.8)
        low = sum(1 for t in tracks if t['confidence'] < 0.6)
        
        # Behavior counts
        hovering = sum(1 for t in tracks if t.get('is_hovering', False))
        zigzag = sum(1 for t in tracks if t.get('is_zigzag', False))
        accelerating = sum(1 for t in tracks if t.get('is_accelerating', False))
        
        # Detection method stats
        method_counts = defaultdict(int)
        for det in detections:
            method_counts[det.get('method', 'unknown')] += 1
        
        # Voxel stats
        active_voxels = np.sum(self.insect_voxels > 0.03)
        high_intensity_voxels = np.sum(self.insect_voxels > 0.5)
        
        # Create info panel
        info_lines = [
            f"FPS: {fps:.1f} | Detections: {len(detections)} | Tracks: {len(tracks)}",
            f"Confidence: VH:{very_high} H:{high} M:{medium} L:{low}",
            f"Behavior: Hover:{hovering} Zigzag:{zigzag} Accel:{accelerating}",
            f"Voxels: {active_voxels:,} ({high_intensity_voxels:,} bright)",
            f"Threshold: {self.min_confidence_display:.1f} | Auto-tune: {self.detector.auto_tune}",
            f"Motion sens: {self.detector.motion_threshold} | Depth range: {self.camera_z:.1f}-{self.camera_z+2:.1f}"
        ]
        
        # Method breakdown
        if method_counts:
            methods = ", ".join([f"{method[:3]}:{count}" for method, count in list(method_counts.items())[:3]])
            info_lines.append(f"Methods: {methods}")
        
        # Draw info panel with background
        panel_height = len(info_lines) * 22 + 10
        cv2.rectangle(frame, (5, 5), (600, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (600, panel_height), (0, 255, 0), 1)
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw FPS indicator
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
                   
    def _show_advanced_help(self):
        """Show comprehensive help."""
        print("\n" + "="*60)
        print("ADVANCED INSECT DETECTOR - COMPREHENSIVE HELP")
        print("="*60)
        print("DETECTION ALGORITHMS:")
        print("  1. Enhanced background subtraction (MOG2 + KNN)")
        print("  2. Multi-scale frame differencing")
        print("  3. Optical flow analysis")
        print("  4. Texture anomaly detection")
        print()
        print("TRACKING FEATURES:")
        print("  • Kalman filter prediction")
        print("  • Behavioral analysis (hover, zigzag, acceleration)")
        print("  • Adaptive parameter tuning")
        print("  • Multi-factor confidence scoring")
        print()
        print("CONTROLS:")
        print("  q/ESC - Quit")
        print("  r     - Reset system")
        print("  a     - Toggle all detections")
        print("  b     - Toggle behavior labels")
        print("  p     - Toggle Kalman prediction")
        print("  =     - Lower confidence threshold")
        print("  -     - Raise confidence threshold")
        print("  t     - Toggle auto-tuning")
        print("  s     - Increase sensitivity")
        print("  d     - Decrease sensitivity")
        print("  h     - Show this help")
        print()
        print("CURRENT SETTINGS:")
        print(f"  Confidence threshold: {self.min_confidence_display:.1f}")
        print(f"  Motion sensitivity: {self.detector.motion_threshold}")
        print(f"  Auto-tuning: {self.detector.auto_tune}")
        print(f"  Show all detections: {self.show_all_detections}")
        print(f"  Show behavior labels: {self.show_behavior_labels}")
        print(f"  Show prediction: {self.show_prediction}")
        print()
        print("COLOR CODING:")
        print("  Red      - Very high confidence (>90%)")
        print("  Yellow   - Hovering behavior")
        print("  Magenta  - Zigzag flight pattern")
        print("  Cyan     - Accelerating movement")
        print("  Orange   - Good confidence (70-90%)")
        print("  Green    - Possible insect")
        print()
        print("VOXEL FEATURES:")
        print("  • High-resolution 3D tracking")
        print("  • Velocity vectors")
        print("  • Adaptive depth estimation")
        print("  • Behavior-based coloring")
        print("="*60)
        
    def _reset_advanced_system(self):
        """Reset advanced system."""
        self.insect_voxels.fill(0)
        self.voxel_colors.fill(0)
        self.voxel_velocities.fill(0)
        self.detector.tracks.clear()
        self.detector.kalman_filters.clear()
        self.detector.next_track_id = 0
        self.detector.frame_buffer.clear()
        self.detection_stats.clear()
        print("Advanced system reset - all data cleared")
        
    def stop(self):
        """Clean up advanced system."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Advanced multi-layer insect detection')
    parser.add_argument('url', nargs='?',
                       default='https://www.youtube.com/watch?v=ipf7ifVSeDU',
                       help='Video URL')
    args = parser.parse_args()
    
    if args.url.isdigit():
        args.url = int(args.url)
        
    print("🔬 Starting Advanced Multi-Layer Insect Detection System")
    print(f"📹 Source: {args.url}")
    print("🚀 Features: 4 detection algorithms, Kalman tracking, behavioral analysis")
    
    tracker = AdvancedInsectVoxelTracker(args.url)
    
    try:
        tracker.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()