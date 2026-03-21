#!/usr/bin/env python3
"""Ultimate insect detection optimized for flying insects in outdoor environments."""

import cv2
import numpy as np
import time
import subprocess
import argparse
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

class UltimateInsectDetector:
    """Ultimate insect detection system optimized for outdoor flying insects."""
    
    def __init__(self):
        # Ultra-specific parameters for flying insects
        self.min_area = 1           # Single pixels for distant insects
        self.max_area = 150         # Larger for close insects/moths
        self.min_perimeter = 3
        
        # Shape filtering optimized for insects
        self.min_aspect_ratio = 0.1
        self.max_aspect_ratio = 10.0
        self.min_solidity = 0.15    # Very loose for irregular insect shapes
        self.min_extent = 0.1
        
        # Motion parameters - ultra-sensitive
        self.motion_threshold = 4
        self.gaussian_blur = 1      # Preserve tiny details
        
        # Multiple background subtractors for different scenarios
        self.bg_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=50,         # Quick adaptation for outdoor scenes
            varThreshold=3,     # Ultra-sensitive
            detectShadows=False
        )
        
        self.bg_knn = cv2.createBackgroundSubtractorKNN(
            history=50,
            dist2Threshold=100,
            detectShadows=False
        )
        
        # Frame differencing buffers
        self.frame_buffer = deque(maxlen=4)
        self.processed_frames = 0
        
        # Tracking with prediction
        self.tracks = {}
        self.next_track_id = 0
        
        # Insect behavior patterns
        self.min_speed = 1.5        # Very slow insects
        self.max_speed = 80.0       # Very fast insects
        self.erratic_threshold = 25 # Direction change degrees
        self.hover_frames = 8       # Frames to detect hovering
        
        # Adaptive thresholding
        self.adaptive_enabled = True
        self.recent_detection_counts = deque(maxlen=60)  # 2 seconds at 30fps
        self.target_detections_per_second = 8
        
    def detect_flying_insects(self, frame: np.ndarray) -> List[Dict]:
        """Comprehensive flying insect detection."""
        detections = []
        self.processed_frames += 1
        
        # Method 1: Dual background subtraction
        bg_detections = self._detect_dual_background_subtraction(frame)
        detections.extend(bg_detections)
        
        # Method 2: Enhanced frame differencing
        if len(self.frame_buffer) >= 2:
            diff_detections = self._detect_enhanced_frame_difference(frame)
            detections.extend(diff_detections)
        
        # Method 3: Motion blob detection
        motion_detections = self._detect_motion_blobs(frame)
        detections.extend(motion_detections)
        
        # Method 4: Edge-based detection for fast insects
        edge_detections = self._detect_edge_motion(frame)
        detections.extend(edge_detections)
        
        # Add frame to buffer
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        # Merge and filter detections
        final_detections = self._merge_filter_detections(detections, frame)
        
        # Adaptive parameter adjustment
        if self.adaptive_enabled:
            self._adapt_sensitivity(len(final_detections))
        
        return final_detections
    
    def _detect_dual_background_subtraction(self, frame: np.ndarray) -> List[Dict]:
        """Dual background subtraction for robust detection."""
        detections = []
        
        # MOG2 - good for regular backgrounds
        fg_mog2 = self.bg_mog2.apply(frame)
        
        # KNN - good for complex backgrounds
        fg_knn = self.bg_knn.apply(frame)
        
        # Combine masks with OR operation
        combined_fg = cv2.bitwise_or(fg_mog2, fg_knn)
        
        # Noise reduction while preserving small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_fg = cv2.morphologyEx(combined_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_fg = cv2.morphologyEx(combined_fg, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(combined_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._analyze_insect_contour(contour, 'background_dual')
            if detection:
                detections.append(detection)
        
        return detections
    
    def _detect_enhanced_frame_difference(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced frame differencing for fast-moving insects."""
        detections = []
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (self.gaussian_blur, self.gaussian_blur), 0)
        
        # Compare with multiple previous frames
        for i, prev_gray in enumerate(list(self.frame_buffer)[-3:]):
            weight = 1.0 / (i + 1)  # Recent frames weighted more
            
            # Frame difference
            diff = cv2.absdiff(current_gray, prev_gray)
            
            # Adaptive threshold based on image content
            mean_intensity = np.mean(diff)
            std_intensity = np.std(diff)
            adaptive_threshold = max(self.motion_threshold, mean_intensity + 0.5 * std_intensity)
            
            _, thresh = cv2.threshold(diff, adaptive_threshold, 255, cv2.THRESH_BINARY)
            
            # Minimal morphology to connect nearby pixels
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                detection = self._analyze_insect_contour(contour, f'frame_diff_{i}')
                if detection:
                    detection['weight'] = weight
                    detections.append(detection)
        
        return detections
    
    def _detect_motion_blobs(self, frame: np.ndarray) -> List[Dict]:
        """Detect motion blobs for medium-sized insects."""
        if len(self.frame_buffer) < 1:
            return []
        
        detections = []
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = self.frame_buffer[-1]
        
        # Absolute difference
        diff = cv2.absdiff(current_gray, prev_gray)
        
        # Apply median filter to reduce noise
        diff = cv2.medianBlur(diff, 3)
        
        # Dynamic threshold based on image statistics
        threshold = max(self.motion_threshold, np.percentile(diff, 95) * 0.3)
        _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._analyze_insect_contour(contour, 'motion_blob')
            if detection:
                detections.append(detection)
        
        return detections
    
    def _detect_edge_motion(self, frame: np.ndarray) -> List[Dict]:
        """Detect fast insects using edge information."""
        if len(self.frame_buffer) < 1:
            return []
        
        detections = []
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = self.frame_buffer[-1]
        
        # Calculate edges
        current_edges = cv2.Canny(current_gray, 30, 100)
        prev_edges = cv2.Canny(prev_gray, 30, 100)
        
        # Edge difference
        edge_diff = cv2.absdiff(current_edges, prev_edges)
        
        # Dilate to connect nearby edge changes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_diff = cv2.dilate(edge_diff, kernel, iterations=1)
        
        # Find contours in edge differences
        contours, _ = cv2.findContours(edge_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._analyze_insect_contour(contour, 'edge_motion')
            if detection:
                detections.append(detection)
        
        return detections
    
    def _analyze_insect_contour(self, contour, method: str) -> Optional[Dict]:
        """Analyze contour to determine if it's an insect."""
        if len(contour) < 3:
            return None
        
        # Basic measurements
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter < self.min_perimeter:
            return None
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        
        # Aspect ratio check
        aspect_ratio = w / h if h > 0 else 0
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return None
        
        # Solidity (how solid vs. sparse the shape is)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < self.min_solidity:
            return None
        
        # Extent (how much of bounding box is filled)
        extent = area / (w * h) if w * h > 0 else 0
        if extent < self.min_extent:
            return None
        
        # Additional insect-specific features
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Eccentricity for elongated insects
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2) if major_axis > 0 else 0
            except:
                eccentricity = 0
        else:
            eccentricity = 0
        
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
            'contour': contour,
            'method': method,
            'timestamp': self.processed_frames
        }
    
    def _merge_filter_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Merge nearby detections and apply final filtering."""
        if not detections:
            return []
        
        # Group nearby detections
        merged_detections = []
        used_indices = set()
        
        for i, det in enumerate(detections):
            if i in used_indices:
                continue
            
            # Find nearby detections (within 10 pixels)
            nearby_detections = [det]
            center_i = det['center']
            
            for j, other_det in enumerate(detections):
                if j <= i or j in used_indices:
                    continue
                
                center_j = other_det['center']
                distance = math.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
                
                if distance < 10:
                    nearby_detections.append(other_det)
                    used_indices.add(j)
            
            # Merge if multiple detections found
            if len(nearby_detections) > 1:
                merged = self._merge_nearby_detections(nearby_detections)
                merged_detections.append(merged)
            else:
                merged_detections.append(det)
            
            used_indices.add(i)
        
        # Apply final quality filters
        final_detections = []
        for detection in merged_detections:
            if self._passes_quality_filter(detection, frame):
                final_detections.append(detection)
        
        return final_detections
    
    def _merge_nearby_detections(self, detections: List[Dict]) -> Dict:
        """Merge multiple nearby detections."""
        # Weight by area for center calculation
        total_area = sum(det['area'] for det in detections)
        
        if total_area > 0:
            center_x = sum(det['center'][0] * det['area'] for det in detections) / total_area
            center_y = sum(det['center'][1] * det['area'] for det in detections) / total_area
        else:
            center_x = np.mean([det['center'][0] for det in detections])
            center_y = np.mean([det['center'][1] for det in detections])
        
        # Take best properties from largest detection
        largest_det = max(detections, key=lambda d: d['area'])
        
        merged = largest_det.copy()
        merged['center'] = (int(center_x), int(center_y))
        merged['area'] = total_area
        merged['method'] = 'merged'
        merged['merge_count'] = len(detections)
        
        return merged
    
    def _passes_quality_filter(self, detection: Dict, frame: np.ndarray) -> bool:
        """Apply final quality filtering."""
        # Size constraints
        area = detection['area']
        if area < 1 or area > 200:
            return False
        
        # Shape constraints
        if detection['compactness'] < 0.05 or detection['compactness'] > 1.2:
            return False
        
        # Boundary check
        center_x, center_y = detection['center']
        if center_x < 5 or center_y < 5 or center_x >= frame.shape[1] - 5 or center_y >= frame.shape[0] - 5:
            return False
        
        # Intensity analysis in detection area
        x, y, w, h = detection['bbox']
        
        # Ensure valid ROI
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check for sufficient contrast
        intensity_range = np.max(gray_roi) - np.min(gray_roi)
        if intensity_range < 10:  # Too uniform
            return False
        
        return True
    
    def _adapt_sensitivity(self, current_detections: int):
        """Adapt detection sensitivity based on recent performance."""
        self.recent_detection_counts.append(current_detections)
        
        if len(self.recent_detection_counts) >= 30:  # 1 second of data
            recent_avg = np.mean(list(self.recent_detection_counts)[-30:])
            
            # Too many detections - reduce sensitivity
            if recent_avg > self.target_detections_per_second * 2:
                self.motion_threshold = min(15, self.motion_threshold + 1)
                self.min_area = min(8, self.min_area + 1)
            
            # Too few detections - increase sensitivity
            elif recent_avg < self.target_detections_per_second * 0.3:
                self.motion_threshold = max(2, self.motion_threshold - 1)
                self.min_area = max(1, self.min_area - 1)
    
    def update_tracks_with_prediction(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with simple prediction."""
        # Simple tracking without Kalman filter
        updated_tracks = {}
        assigned_detections = set()
        
        # Update existing tracks
        for track_id, track in self.tracks.items():
            best_detection = None
            best_distance = float('inf')
            best_idx = -1
            
            # Predict next position based on velocity
            if len(track['positions']) >= 2:
                last_pos = track['positions'][-1]
                prev_pos = track['positions'][-2]
                velocity = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
                predicted_pos = (last_pos[0] + velocity[0], last_pos[1] + velocity[1])
            else:
                predicted_pos = track['last_center']
            
            # Find closest detection to prediction
            for i, detection in enumerate(detections):
                if i in assigned_detections:
                    continue
                
                det_center = detection['center']
                distance = math.sqrt((det_center[0] - predicted_pos[0])**2 + 
                                   (det_center[1] - predicted_pos[1])**2)
                
                # Dynamic search radius based on track speed
                search_radius = min(40, max(15, track.get('speed', 5) * 2))
                
                if distance < best_distance and distance < search_radius:
                    best_distance = distance
                    best_detection = detection
                    best_idx = i
            
            if best_detection:
                # Update track
                track['positions'].append(best_detection['center'])
                track['last_center'] = best_detection['center']
                track['lost_frames'] = 0
                track['detection'] = best_detection
                track['age'] += 1
                
                # Analyze behavior
                self._analyze_insect_behavior(track)
                
                updated_tracks[track_id] = track
                assigned_detections.add(best_idx)
            else:
                # Track lost
                track['lost_frames'] += 1
                if track['lost_frames'] < 8:  # Keep lost tracks briefly
                    updated_tracks[track_id] = track
        
        # Create new tracks
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                new_track = {
                    'id': self.next_track_id,
                    'positions': [detection['center']],
                    'last_center': detection['center'],
                    'lost_frames': 0,
                    'detection': detection,
                    'age': 1,
                    'speed': 0,
                    'direction_changes': 0,
                    'is_hovering': False,
                    'is_fast': False,
                    'is_erratic': False,
                    'confidence': 0.4
                }
                updated_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        self.tracks = updated_tracks
        
        # Return active tracks with reasonable confidence
        return [track for track in self.tracks.values() 
                if track['lost_frames'] == 0 and track['confidence'] > 0.3]
    
    def _analyze_insect_behavior(self, track: Dict):
        """Analyze track for insect-like behavior patterns."""
        positions = track['positions']
        
        if len(positions) < 2:
            return
        
        # Calculate speed
        if len(positions) >= 2:
            recent_positions = positions[-min(8, len(positions)):]
            total_distance = 0
            
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            
            track['speed'] = total_distance / (len(recent_positions) - 1) if len(recent_positions) > 1 else 0
        
        # Analyze direction changes (erratic flight)
        if len(positions) >= 4:
            recent_pos = positions[-min(6, len(positions)):]
            direction_changes = 0
            
            for i in range(2, len(recent_pos)):
                # Calculate direction vectors
                v1 = (recent_pos[i-1][0] - recent_pos[i-2][0], recent_pos[i-1][1] - recent_pos[i-2][1])
                v2 = (recent_pos[i][0] - recent_pos[i-1][0], recent_pos[i][1] - recent_pos[i-1][1])
                
                # Calculate angle between vectors
                if (v1[0] != 0 or v1[1] != 0) and (v2[0] != 0 or v2[1] != 0):
                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                        angle_deg = math.degrees(math.acos(cos_angle))
                        
                        if angle_deg > self.erratic_threshold:
                            direction_changes += 1
            
            track['direction_changes'] = direction_changes
            track['is_erratic'] = direction_changes > 1
        
        # Detect hovering
        if len(positions) >= self.hover_frames:
            recent_pos = positions[-self.hover_frames:]
            center_x = sum(pos[0] for pos in recent_pos) / len(recent_pos)
            center_y = sum(pos[1] for pos in recent_pos) / len(recent_pos)
            
            max_distance = max(math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
                             for pos in recent_pos)
            track['is_hovering'] = max_distance < 6
        
        # Detect fast movement
        track['is_fast'] = track['speed'] > 20
        
        # Calculate confidence
        confidence = 0.4
        
        # Size preference (smaller more likely insects in outdoor video)
        area = track['detection']['area']
        if 1 <= area <= 15:
            confidence += 0.3
        elif 15 < area <= 40:
            confidence += 0.2
        elif 40 < area <= 80:
            confidence += 0.1
        
        # Speed bonus
        if self.min_speed <= track['speed'] <= self.max_speed:
            confidence += 0.2
        
        # Behavior bonuses
        if track['is_erratic']:
            confidence += 0.15
        if track['is_hovering']:
            confidence += 0.1
        if track['is_fast']:
            confidence += 0.1
        
        # Age bonus
        if track['age'] > 5:
            confidence += 0.1
        
        # Method bonus
        method = track['detection'].get('method', '')
        if 'merged' in method:
            confidence += 0.1
        
        track['confidence'] = min(1.0, confidence)


class UltimateInsectTracker:
    """Ultimate insect tracking with 3D voxel visualization."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.detector = UltimateInsectDetector()
        
        # Video parameters
        self.cap = None
        self.width = 1024  # Higher resolution
        self.height = 768
        
        # Ultra-high resolution voxel system
        self.voxel_dims = (150, 150, 75)
        self.voxel_size = 0.01  # Very fine voxels
        self.insect_voxels = np.zeros(self.voxel_dims, dtype=np.float32)
        self.voxel_colors = np.zeros((*self.voxel_dims, 3), dtype=np.float32)
        self.voxel_decay = 0.91  # Quick decay for dynamic scenes
        
        # Camera parameters
        self.camera_fov = 80  # Wide field of view for outdoor scenes
        self.camera_z = 0.5
        
        # Display settings
        self.show_all_detections = False
        self.min_confidence_display = 0.5
        self.show_trails = True
        self.show_behavior = True
        self.show_stats = True
        
        # Performance
        self.frame_times = deque(maxlen=60)
        self.total_insects_detected = 0
        
    def get_stream_url(self):
        """Get high-quality stream URL."""
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            try:
                # Try for highest quality
                cmd = ['yt-dlp', '-f', 'best[height<=1080]', '--get-url', self.stream_url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        return self.stream_url
        
    def start(self):
        """Start ultimate insect detection."""
        url = self.get_stream_url()
        print(f"🦟 Ultimate Insect Detection: {url}")
        
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open stream")
        
        print("🚀 Ultimate Insect Detection System Started!")
        print("\n🎯 Optimized for:")
        print("  • Flying insects in outdoor environments")
        print("  • Ultra-sensitive detection (1-150 pixels)")
        print("  • 4-layer detection algorithms")
        print("  • Adaptive sensitivity")
        print("  • Behavioral analysis")
        print("\n🎮 Controls:")
        print("  q/ESC - Quit")
        print("  r - Reset")
        print("  a - Toggle all detections")
        print("  t - Toggle trails")
        print("  b - Toggle behavior labels")
        print("  s - Toggle statistics")
        print("  = - Lower confidence threshold")
        print("  - - Raise confidence threshold")
        print("  space - Pause/Resume")
        print("  h - Help")
        
        cv2.namedWindow('Ultimate Insect Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ultimate Insect Detector', 1400, 900)
        
        paused = False
        frame_count = 0
        
        while True:
            frame_start = time.time()
            
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of stream")
                    break
                
                frame_count += 1
                
                # Resize maintaining aspect ratio
                frame = cv2.resize(frame, (self.width, self.height))
                
                # Ultimate detection
                detections = self.detector.detect_flying_insects(frame)
                
                # Advanced tracking
                active_tracks = self.detector.update_tracks_with_prediction(detections)
                
                # Update voxels
                self._update_ultimate_voxels(active_tracks)
                
                # Count insects
                confident_tracks = [t for t in active_tracks if t['confidence'] >= self.min_confidence_display]
                self.total_insects_detected += len(confident_tracks)
            
            # Create visualization
            vis_frame = self._create_ultimate_visualization(frame, detections if not paused else [], active_tracks if not paused else [])
            
            cv2.imshow('Ultimate Insect Detector', vis_frame)
            
            # Performance tracking
            if not paused:
                self.frame_times.append(time.time() - frame_start)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):  # Space for pause
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('r'):
                self._reset_ultimate_system()
            elif key == ord('a'):
                self.show_all_detections = not self.show_all_detections
                print(f"Show all detections: {self.show_all_detections}")
            elif key == ord('t'):
                self.show_trails = not self.show_trails
                print(f"Show trails: {self.show_trails}")
            elif key == ord('b'):
                self.show_behavior = not self.show_behavior
                print(f"Show behavior: {self.show_behavior}")
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print(f"Show statistics: {self.show_stats}")
            elif key == ord('=') or key == ord('+'):
                self.min_confidence_display = max(0.1, self.min_confidence_display - 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('-'):
                self.min_confidence_display = min(0.9, self.min_confidence_display + 0.1)
                print(f"Confidence threshold: {self.min_confidence_display:.1f}")
            elif key == ord('h'):
                self._show_ultimate_help()
                
    def _update_ultimate_voxels(self, tracks: List[Dict]):
        """Update ultra-high resolution voxels."""
        # Apply decay
        self.insect_voxels *= self.voxel_decay
        self.voxel_colors *= self.voxel_decay
        
        for track in tracks:
            if track['confidence'] < 0.4:
                continue
                
            center_x, center_y = track['last_center']
            
            # Advanced depth estimation
            area = track['detection']['area']
            speed = track['speed']
            age = track['age']
            
            # Multi-factor depth estimation
            # Smaller objects assumed further away
            size_factor = max(0.2, min(2.0, 25.0 / (area + 2)))
            # Faster objects assumed closer
            speed_factor = max(0.5, min(1.5, (speed + 5) / 15.0))
            # Older tracks more reliable depth
            age_factor = min(1.2, 1.0 + age * 0.02)
            
            estimated_depth = size_factor * speed_factor * age_factor
            
            # Convert to world coordinates
            norm_x = (center_x / self.width - 0.5) * 2.0
            norm_y = (center_y / self.height - 0.5) * 2.0
            
            fov_scale = math.tan(math.radians(self.camera_fov / 2))
            world_x = norm_x * fov_scale * estimated_depth
            world_y = -norm_y * fov_scale * estimated_depth
            world_z = self.camera_z + estimated_depth
            
            # Convert to voxel indices
            vx = int(world_x / self.voxel_size + self.voxel_dims[0] / 2)
            vy = int(world_y / self.voxel_size + self.voxel_dims[1] / 2)
            vz = int(world_z / self.voxel_size)
            
            if (0 <= vx < self.voxel_dims[0] and 
                0 <= vy < self.voxel_dims[1] and 
                0 <= vz < self.voxel_dims[2]):
                
                # Color based on behavior and confidence
                if track['confidence'] > 0.9:
                    color = np.array([1.0, 0.0, 0.0])  # Red - very confident
                elif track['is_hovering']:
                    color = np.array([1.0, 1.0, 0.0])  # Yellow - hovering
                elif track['is_erratic']:
                    color = np.array([1.0, 0.0, 1.0])  # Magenta - erratic
                elif track['is_fast']:
                    color = np.array([0.0, 1.0, 1.0])  # Cyan - fast
                else:
                    color = np.array([0.0, 1.0, 0.0])  # Green - normal
                
                intensity = track['confidence'] * 2.0
                
                # High-resolution voxel update with fine spreading
                spread_radius = max(1, int(track['confidence'] * 2))
                for dx in range(-spread_radius, spread_radius + 1):
                    for dy in range(-spread_radius, spread_radius + 1):
                        for dz in range(-1, 2):
                            nvx, nvy, nvz = vx + dx, vy + dy, vz + dz
                            if (0 <= nvx < self.voxel_dims[0] and 
                                0 <= nvy < self.voxel_dims[1] and 
                                0 <= nvz < self.voxel_dims[2]):
                                
                                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                                falloff = max(0.1, 1.0 / (1.0 + distance * 0.5))
                                
                                update_amount = intensity * falloff
                                self.insect_voxels[nvx, nvy, nvz] += update_amount
                                self.voxel_colors[nvx, nvy, nvz] += color * update_amount
                                
    def _create_ultimate_visualization(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]) -> np.ndarray:
        """Create ultimate visualization."""
        vis = frame.copy()
        
        # Filter tracks for display
        if self.show_all_detections:
            display_tracks = tracks
        else:
            display_tracks = [t for t in tracks if t['confidence'] >= self.min_confidence_display]
        
        # Draw detection boxes
        for track in display_tracks:
            detection = track['detection']
            x, y, w, h = detection['bbox']
            center_x, center_y = track['last_center']
            confidence = track['confidence']
            
            # Dynamic color and thickness
            if confidence > 0.9:
                color = (0, 0, 255)    # Red - very confident
                thickness = 3
            elif track.get('is_hovering'):
                color = (0, 255, 255)  # Yellow - hovering
                thickness = 2
            elif track.get('is_erratic'):
                color = (255, 0, 255)  # Magenta - erratic
                thickness = 2
            elif track.get('is_fast'):
                color = (255, 255, 0)  # Cyan - fast
                thickness = 2
            elif confidence > 0.7:
                color = (0, 165, 255)  # Orange - good
                thickness = 2
            else:
                color = (0, 255, 0)    # Green - possible
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            cv2.circle(vis, (center_x, center_y), 2, color, -1)
            
            # Labels
            label_parts = [f"{confidence:.2f}"]
            if self.show_behavior:
                if track.get('is_hovering'):
                    label_parts.append("HOVER")
                if track.get('is_erratic'):
                    label_parts.append("ERRATIC")
                if track.get('is_fast'):
                    label_parts.append("FAST")
            
            label = " ".join(label_parts)
            cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Track ID for high confidence
            if confidence > 0.8:
                cv2.putText(vis, f"#{track['id']}", (x + w + 2, y + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw trails
        if self.show_trails:
            for track in display_tracks:
                if track['confidence'] > 0.6:
                    positions = track['positions'][-20:]  # Last 20 positions
                    if len(positions) > 1:
                        # Trail color based on behavior
                        if track.get('is_hovering'):
                            trail_color = (100, 255, 255)
                        elif track.get('is_erratic'):
                            trail_color = (200, 100, 255)
                        elif track.get('is_fast'):
                            trail_color = (255, 255, 100)
                        else:
                            trail_color = (150, 255, 150)
                        
                        pts = np.array(positions, np.int32)
                        cv2.polylines(vis, [pts], False, trail_color, 2)
        
        # Draw voxel overlay
        self._draw_ultimate_voxel_overlay(vis)
        
        # Add comprehensive stats
        if self.show_stats:
            self._add_ultimate_stats(vis, detections, tracks)
        
        return vis
        
    def _draw_ultimate_voxel_overlay(self, frame: np.ndarray):
        """Draw ultra-high resolution voxel overlay."""
        active_mask = self.insect_voxels > 0.05
        
        if not np.any(active_mask):
            return
        
        coords = np.argwhere(active_mask)
        intensities = self.insect_voxels[active_mask]
        colors = self.voxel_colors[active_mask]
        
        # Normalize colors
        colors = colors / (intensities[:, np.newaxis] + 0.001)
        colors = np.clip(colors, 0, 1)
        
        # Limit for performance but allow more for high resolution
        if len(coords) > 3000:
            indices = np.argsort(intensities)[-3000:]
            coords = coords[indices]
            intensities = intensities[indices]
            colors = colors[indices]
        
        for coord, intensity, color in zip(coords, intensities, colors):
            # Project to screen
            world_x = (coord[0] - self.voxel_dims[0] / 2) * self.voxel_size
            world_y = (coord[1] - self.voxel_dims[1] / 2) * self.voxel_size
            world_z = coord[2] * self.voxel_size
            
            if world_z > 0.05:
                distance = world_z - self.camera_z
                if distance > 0:
                    fov_scale = math.tan(math.radians(self.camera_fov / 2))
                    screen_x = int((world_x / (distance * fov_scale) + 1) * self.width / 2)
                    screen_y = int((-world_y / (distance * fov_scale) + 1) * self.height / 2)
                    
                    if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                        # Ultra-fine voxel rendering
                        size = max(1, int(5 * intensity / distance))
                        bgr_color = tuple(map(int, color[::-1] * 255))
                        
                        # Multi-layer glow effect
                        for r in range(size + 3, 0, -1):
                            alpha = 0.2 * (r / (size + 3)) * intensity
                            glow_color = tuple(int(c * alpha) for c in bgr_color)
                            cv2.circle(frame, (screen_x, screen_y), r, glow_color, -1)
                        
                        # Bright core
                        cv2.circle(frame, (screen_x, screen_y), max(1, size), bgr_color, -1)
                        
    def _add_ultimate_stats(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]):
        """Add comprehensive statistics display."""
        # Performance
        fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        
        # Confidence distribution
        very_high = sum(1 for t in tracks if t['confidence'] > 0.9)
        high = sum(1 for t in tracks if 0.8 <= t['confidence'] <= 0.9)
        medium = sum(1 for t in tracks if 0.6 <= t['confidence'] < 0.8)
        low = sum(1 for t in tracks if 0.4 <= t['confidence'] < 0.6)
        
        # Behavior analysis
        hovering = sum(1 for t in tracks if t.get('is_hovering', False))
        erratic = sum(1 for t in tracks if t.get('is_erratic', False))
        fast = sum(1 for t in tracks if t.get('is_fast', False))
        
        # Detection methods
        method_counts = defaultdict(int)
        for det in detections:
            method_counts[det.get('method', 'unknown')] += 1
        
        # Voxel statistics
        total_voxels = np.sum(self.insect_voxels > 0.01)
        bright_voxels = np.sum(self.insect_voxels > 0.5)
        max_intensity = np.max(self.insect_voxels)
        
        # Create stats panel
        stats_lines = [
            f"FPS: {fps:.1f} | Frame: {self.detector.processed_frames}",
            f"Detections: {len(detections)} | Active tracks: {len(tracks)}",
            f"Confidence: VH:{very_high} H:{high} M:{medium} L:{low}",
            f"Behavior: Hover:{hovering} Erratic:{erratic} Fast:{fast}",
            f"Voxels: {total_voxels:,} ({bright_voxels:,} bright)",
            f"Sensitivity: {self.detector.motion_threshold} | Min area: {self.detector.min_area}",
            f"Threshold: {self.min_confidence_display:.1f} | Total found: {self.total_insects_detected}"
        ]
        
        # Detection methods breakdown
        if method_counts:
            methods_str = " | ".join([f"{k[:8]}:{v}" for k, v in list(method_counts.items())[:4]])
            stats_lines.append(f"Methods: {methods_str}")
        
        # Create stats background
        panel_width = 650
        panel_height = len(stats_lines) * 25 + 15
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 255, 0), 2)
        
        # Draw stats text
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (15, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # FPS indicator
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
        
    def _show_ultimate_help(self):
        """Show comprehensive help."""
        print("\n" + "="*70)
        print("🦟 ULTIMATE INSECT DETECTOR - HELP")
        print("="*70)
        print("🎯 DETECTION FEATURES:")
        print("  • 4-layer detection: Background subtraction, frame diff, motion blobs, edge motion")
        print("  • Ultra-sensitive: 1-150 pixel detection range")
        print("  • Adaptive sensitivity: Auto-adjusts based on detection rate")
        print("  • Behavioral analysis: Hovering, erratic flight, speed analysis")
        print("  • High-resolution voxels: 150x150x75 grid with 0.01m precision")
        print()
        print("🎮 CONTROLS:")
        print("  q/ESC    - Quit application")
        print("  r        - Reset system (voxels, tracks, statistics)")
        print("  space    - Pause/Resume processing")
        print("  a        - Toggle show all detections vs confident only")
        print("  t        - Toggle track trails")
        print("  b        - Toggle behavior labels")
        print("  s        - Toggle statistics panel")
        print("  =        - Lower confidence threshold (show more)")
        print("  -        - Raise confidence threshold (show fewer)")
        print("  h        - Show this help")
        print()
        print("🎨 COLOR CODING:")
        print("  Red      - Very high confidence (>90%)")
        print("  Yellow   - Hovering insects")
        print("  Magenta  - Erratic flight patterns")
        print("  Cyan     - Fast-moving insects")
        print("  Orange   - Good confidence (70-90%)")
        print("  Green    - Possible insects")
        print()
        print("📊 CURRENT SETTINGS:")
        print(f"  Confidence threshold: {self.min_confidence_display:.1f}")
        print(f"  Motion sensitivity: {self.detector.motion_threshold}")
        print(f"  Min detection area: {self.detector.min_area} pixels")
        print(f"  Adaptive mode: {self.detector.adaptive_enabled}")
        print(f"  Show all detections: {self.show_all_detections}")
        print(f"  Show trails: {self.show_trails}")
        print(f"  Show behavior: {self.show_behavior}")
        print(f"  Show statistics: {self.show_stats}")
        print("="*70)
        
    def _reset_ultimate_system(self):
        """Reset everything."""
        self.insect_voxels.fill(0)
        self.voxel_colors.fill(0)
        self.detector.tracks.clear()
        self.detector.next_track_id = 0
        self.detector.frame_buffer.clear()
        self.detector.processed_frames = 0
        self.total_insects_detected = 0
        print("🔄 Ultimate system reset - all data cleared")
        
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Ultimate insect detection system')
    parser.add_argument('url', nargs='?',
                       default='https://www.youtube.com/watch?v=ZdfJcN-yBIE',
                       help='Video URL')
    args = parser.parse_args()
    
    if args.url.isdigit():
        args.url = int(args.url)
        
    print("🦟 Ultimate Insect Detection System")
    print(f"📹 Processing: {args.url}")
    
    tracker = UltimateInsectTracker(args.url)
    
    try:
        tracker.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()